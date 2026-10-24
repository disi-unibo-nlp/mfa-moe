"""Reduce the raw capture of one trace to what is stored: a JSON packet and a tensor file.

The raw capture is a dict of layers x tokens tensors (router signal, hidden states, selected expert ids, their weights,
selection scores). The packet holds the trace, the per-layer features, the sliding-window routing dynamics, the
class/position views of every cohort that selected this trace and the expert counts. The tensor file holds the ids and
weights (fp16) and the selection margins.
"""
import copy
import gzip
import hashlib
import json
import re
from pathlib import Path

import torch

from capture.dynamics import SCHEMA_VERSION
from capture.dynamics.classes import saved_layout, sentence_table
from capture.dynamics.common import DEFAULT_CONFIG
from capture.dynamics.routing import probabilities, reduce_arrays
from capture.features import FEATURE_SCHEMA_VERSION, compute_layer_features, json_safe_features
from capture.manifest import digest, read_location
from capture.native_selection import require_native_boundary
from capture.sentence_schema import infer_schema, install
from capture.views import compute_views
from labeling import spans
from labeling.spans import selected_sentence_indices, trace_digest, validate_annotation
from schemas import TraceRecord
from store import CLASSES

# The features every capture must have for every routed layer.
LEGACY = ('router_entropy', 'router_confidence', 'router_margin', 'router_selected_mass',
          'router_boundary_margin', 'router_topk_non_topk_gap', 'router_switch_rate',
          'router_topk_overlap', 'hidden_norm', 'hidden_step_distance', 'hidden_router_geometry')


def load_annotation(cohort):
    """The labels frozen into the manifest (inline) or in a gzip file, checked against their hash."""
    info = cohort['annotation']
    if 'inline' in info:
        if digest(info['inline']) != info.get('sha256'):
            raise ValueError('Inline annotation changed')
        # A fresh copy, so that normalising it cannot change the manifest.
        return json.loads(json.dumps(info['inline'], allow_nan=False))
    raw = gzip.decompress(Path(info['path']).read_bytes())
    if hashlib.sha256(raw).hexdigest() != info['sha256']:
        raise ValueError('Annotation changed')
    return json.loads(raw)


def normalize_annotation(trace, annotation):
    """Represent unlabeled sentences explicitly; never invent or overwrite labels."""
    install()
    infer_schema(trace, annotation)
    annotation = copy.deepcopy(annotation)
    # Compact batch joins omit this field; the content-bound source retains it.
    if 'sentence_selection' not in annotation:
        annotation['sentence_selection'] = copy.deepcopy(trace.metadata.get('sentence_selection'))
    units = spans.sentence_spans(trace)  # looked up now: install() replaces it with the schema-aware version
    expected = [units[i] for i in selected_sentence_indices(trace, units)]
    if annotation.get('sentence_selection') != trace.metadata.get('sentence_selection'):
        raise ValueError('Annotation selection mismatch')
    by_index = {u['index']: u for u in annotation['units']}
    if len(by_index) != len(annotation['units']) or set(by_index) - {u['index'] for u in expected}:
        raise ValueError('Invalid annotation unit identities')
    normalized = []
    for unit in expected:
        saved = by_index.get(unit['index'])
        if saved is not None and any(saved.get(k) != v for k, v in unit.items()):
            raise ValueError('Annotation offsets/text changed')
        saved = copy.deepcopy(saved) if saved is not None else dict(unit, status='unknown')
        if saved.get('status') == 'unknown' or saved.get('label') is None:
            saved.pop('label', None)
            saved['status'] = 'unknown'
            saved.setdefault('raw_completion', None)
            saved.setdefault('failure', dict(
                kind='unavailable_source_annotation', error_type='MissingLabelMetadata',
                message='No canonical label in the saved source cohort; original checkpoint remains unchanged',
                finish_reason=None))
        normalized.append(saved)
    annotation['units'] = normalized
    validate_annotation(trace, annotation)
    return annotation


def counts_for_tokens(ids, indices, experts):
    """Per layer, how often each expert was selected on the tokens `indices`."""
    return [torch.bincount(layer[indices].long().reshape(-1), minlength=experts).tolist() for layer in ids]


def expert_class_counts(trace, annotation, ids, experts):
    sentences = sentence_table(trace, annotation, saved_layout(trace))
    groups = {c: set() for c in CLASSES}
    for sentence in sentences:
        if sentence['label'] in groups and sentence['token_indices'] is not None:
            groups[sentence['label']].update(sentence['token_indices'])
    all_tokens = sorted(set().union(*groups.values()))
    return dict(population='labeled reasoning tokens; class gaps excluded', tokens=len(all_tokens),
                counts=counts_for_tokens(ids, all_tokens, experts),
                classes={c: dict(tokens=len(ix), counts=counts_for_tokens(ids, sorted(ix), experts))
                         for c, ix in groups.items()})


def marker_counts(trace, ids, experts):
    offsets = trace.metadata['token_replay']['completion_offsets']
    out = {}
    for marker in ('<think>', '</think>', 'Alternatively'):
        found = [m.span() for m in re.finditer(re.escape(marker), trace.cot_text)]
        indices = [i for i, (a, b) in enumerate(offsets) if b > a and any(b > x and a < y for x, y in found)]
        out[marker] = dict(occurrences=len(found), aligned_tokens=len(indices),
                           counts=counts_for_tokens(ids, indices, experts),
                           status='observed' if found else 'unavailable_no_occurrence')
    return out


def reduce_capture(row, data, backend, manifest):
    spec = manifest['spec']
    layers = spec['layers']
    router, hidden, ids, weights, scores = data['router'], data['hidden'], data['ids'], data['weights'], data['selection']
    experts = router.shape[-1]
    if tuple(ids.shape) != (len(layers), row['completion_tokens'], spec['top_k']):
        raise ValueError('Incomplete dispatched IDs')
    if hidden.shape[:2] != ids.shape[:2] or weights.shape != ids.shape:
        raise ValueError('Incomplete native signals')
    gaps = require_native_boundary(scores, ids)
    values = json_safe_features(compute_layer_features(router, hidden, ids, max_geometry_tokens=128, layer_indices=layers))
    if any(values.get(f'{metric}_l{layer:02d}') is None for layer in layers for metric in LEGACY):
        raise ValueError('Missing legacy metric')
    features = dict(schema_version=FEATURE_SCHEMA_VERSION, config={'max_geometry_tokens': 128},
                    token_count=row['completion_tokens'], layer_indices=layers, num_layers=len(layers),
                    num_experts=experts, top_k=ids.shape[-1], values=values)
    first = TraceRecord(**read_location(row['cohorts'][0]['locations'][0]))
    layout = saved_layout(first)

    class Probabilities:
        shape = router.shape

        def __getitem__(self, index):
            return probabilities(router[index].float().numpy())

    semantics = {**data['semantics'], 'num_experts': experts, 'deterministic_topk': True}
    windows = reduce_arrays(ids, layers=layers, model=first.model_id, probabilities_array=Probabilities(),
                            weights=weights, selection_scores=scores, semantics=semantics,
                            token_indices=layout['reasoning_tokens'])
    if any(w['boundary_gap_mean'] is None for w in windows):
        raise ValueError('Missing native boundary gaps')
    variants = {}
    for cohort in row['cohorts']:
        trace = TraceRecord(**read_location(cohort['locations'][0]))
        annotation = normalize_annotation(trace, load_annotation(cohort)) if cohort.get('annotation') else None
        if trace_digest(trace) != row['trace_sha256'] or saved_layout(trace)['reasoning_tokens'] != layout['reasoning_tokens']:
            raise ValueError('Cohort changes replay or reasoning scope')
        trace.metadata.pop('reasoning_annotation', None)
        if annotation is not None:
            validate_annotation(trace, annotation)
            trace.metadata['reasoning_annotation'] = annotation
        views = compute_views(trace, backend.tokenizer, router, hidden, ids, layers,
                              modes=['full', 'position'] + (['class'] if annotation is not None else []),
                              reference=manifest['position_reference'], max_geometry_tokens=128)
        variants[cohort['id']] = dict(
            source_trace=trace.model_dump(mode='json'),
            original_annotation_sha256=cohort.get('annotation', {}).get('sha256'),
            annotation=annotation, views=views,
            expert_class=expert_class_counts(trace, annotation, ids, experts) if annotation is not None else None)
    first.metadata['forward_provenance'] = backend.provenance
    first.metadata['correlation_features'] = features
    first.metadata['routing_dynamics'] = dict(
        schema_version=SCHEMA_VERSION, config=dict(DEFAULT_CONFIG), trace_sha256=row['trace_sha256'],
        annotation_sha256=None, forward_provenance=backend.provenance, semantics=semantics, windows=windows)
    first.model_logs.layer_indices = layers
    # Outcomes are joined at analysis time from the explicit sidecar; the packet
    # carries only native capture provenance and never a planning-time label.
    packet = dict(schema_version=2, weights_storage_dtype='float16', id=row['id'],
                  trace=first.model_dump(mode='json'), cohorts=variants, native_dispatch_audit=data['audit'],
                  marker_counts=marker_counts(first, ids, experts),
                  native_expert_counts=counts_for_tokens(ids, list(range(ids.shape[1])), experts),
                  manifest_binding=manifest['binding'])
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights > torch.finfo(torch.float16).max).any():
        raise ValueError('Native weights cannot be stored as finite nonnegative fp16')
    tensors = dict(ids=ids.to(torch.int16), weights=weights.to(torch.float16), native_boundary_gaps=gaps.float(),
                   reasoning_tokens=torch.tensor(layout['reasoning_tokens'], dtype=torch.int32))
    return packet, tensors
