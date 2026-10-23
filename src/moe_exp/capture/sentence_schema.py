"""Version-aware segmentation preserving exact saved historical annotations."""
def install():
    import sys
    from moe_exp.labeling import spans
    if getattr(spans.sentence_spans,'_measurement_versioned',False):return
    current=spans.sentence_spans;legacy=spans.sentence_spans_v1
    def versioned(trace):
        version=trace.metadata.get('measurement_sentence_schema',2)
        if version not in (1,2):raise ValueError('Unknown bound sentence schema')
        return legacy(trace) if version==1 else current(trace)
    versioned._measurement_versioned=True
    spans.sentence_spans=versioned
    # Existing imports, if any, must use the same trace-specific segmentation.
    for name in ('moe_exp.correlation_pipeline.dynamics.classes',):
        module=sys.modules.get(name)
        if module is not None:module.sentence_spans=versioned

def infer_schema(trace, annotation):
    from moe_exp.labeling import spans
    units=annotation['units']
    indices=[u['index'] for u in units]
    if len(set(indices))!=len(indices):raise ValueError('Duplicate saved annotation indices')
    trace.metadata.pop('measurement_sentence_schema',None)
    for version,split in ((2,spans.sentence_spans),(1,spans.sentence_spans_v1)):
        expected=split(trace)
        if all(type(u['index']) is int and 0<=u['index']<len(expected) and
               all(u.get(k)==v for k,v in expected[u['index']].items()) for u in units):
            trace.metadata['measurement_sentence_schema']=version
            return version
    raise ValueError('Saved annotation offsets/text match neither exact current nor frozen legacy spans')
