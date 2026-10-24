"""Memory budget, in-memory buffers and backend checks of the vLLM capture.

The buffers hold the signals of one trace (ids, weights, router, selection, hidden) at native precision until the
two tensor-parallel ranks have compared hashes; only the on-disk mixture weights are rounded to fp16, after
finite/range checks.
"""
from __future__ import annotations

from pathlib import Path

from store import WORK

MIB = 1024 ** 2
GIB = 1024 ** 3
MAX_WORKER_BYTES = 100 * GIB  # Two TP workers plus engine/headroom in a 240G job.
MAX_TRANSIENT_BYTES = 96 * GIB
# Where captures may be written: the scratch capture root, the job folders and the capture folder of the project.
# store.private is not used: it accepts all of WORK and HOME, far wider than a capture should be allowed to write.
CAPTURE_ROOTS = (Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3'),
                 WORK / 'mfa-moe' / 'r', WORK / 'mfa-moe' / 'campaign-v3')


def approved_capture_path(output):
    """The resolved folder, if it is under a capture root and not inside a git checkout (captures are never committed)."""
    path = Path(output).resolve()
    if not any(path.is_relative_to(root) for root in CAPTURE_ROOTS):
        raise ValueError('Unapproved capture path: ' + str(path))
    if any((folder / '.git').exists() for folder in (path, *path.parents)):
        raise ValueError('Capture path inside a git checkout: ' + str(path))
    return path


def capture_memory_budget(tokens, layers, hidden_size, num_experts, top_k, *, rank=0):
    """Conservative bytes, including buffers, coverage and serialization headroom."""
    dimensions = (tokens, layers, hidden_size, num_experts, top_k)
    if any(type(x) is not int or x <= 0 for x in dimensions) or top_k > num_experts:
        raise ValueError('Invalid native capture dimensions')
    if rank not in (0, 1):
        raise ValueError('Invalid tensor-parallel rank')
    # IDs may be int64; hidden up to float32. Native weights are kept float32.
    native = tokens * layers * (8 * top_k + 4 * top_k + 8 * num_experts + (4 * hidden_size if rank == 0 else 0))
    coverage = tokens * layers * (5 if rank == 0 else 4)
    output = tokens * layers * (8 * top_k + 2 * top_k + 8 * num_experts + 4 * hidden_size) + 8 * MIB
    # Rounded weights are the only additional full tensor during serialization.
    worker = native + coverage + tokens * layers * top_k * 2 + 64 * MIB
    rank0_native = tokens * layers * (12 * top_k + 8 * num_experts + 4 * hidden_size)
    rank1_native = tokens * layers * (12 * top_k + 8 * num_experts)
    # Snapshot release precedes parent torch.load; leave 64GiB for engines,
    # short HF reference, transient chunks and orchestration. No concurrent full
    # parent tensor copy is budgeted while both worker buffers remain live.
    host = rank0_native + rank1_native + tokens * layers * (9 + 2 * top_k) + 64 * GIB
    if worker > MAX_WORKER_BYTES or output > MAX_TRANSIENT_BYTES or host > 240 * GIB:
        raise ValueError('Native capture exceeds 240G job memory envelope')
    return {'worker_bytes': worker, 'transient_bytes': output, 'native_tensor_bytes': native, 'host_peak_budget_bytes': host}


class CaptureBuffers:
    def __init__(self, layers, tokens, *, rank=0, max_bytes=MAX_WORKER_BYTES):
        if not layers or list(layers) != sorted(set(layers)) or type(tokens) is not int or tokens <= 0:
            raise ValueError('Invalid layer/token coverage contract')
        if rank not in (0, 1) or not 0 < max_bytes <= MAX_WORKER_BYTES:
            raise ValueError('Invalid native buffer memory cap')
        self.layers = list(layers)
        self.indices = {layer: i for i, layer in enumerate(layers)}
        self.tokens, self.rank, self.max_bytes = tokens, rank, max_bytes
        self.arrays, self.coverage = {}, {}
        self.allocated_bytes = 0
        self.expected = {'ids', 'weights', 'router', 'selection'} | ({'hidden'} if rank == 0 else set())

    def write(self, layer, positions, values):
        import torch
        if layer not in self.indices or not values or not set(values) <= self.expected:
            raise ValueError('Unexpected capture layer or signal')
        positions = torch.as_tensor(positions, dtype=torch.long, device='cpu')
        if positions.ndim != 1 or not len(positions) or bool(((positions < 0) | (positions >= self.tokens)).any()):
            raise ValueError('Capture position outside completion')
        if len(positions.unique()) != len(positions):
            raise ValueError('Duplicate positions within one capture batch')
        index = self.indices[layer]
        for name, value in values.items():
            value = value.detach().to(device='cpu')
            if value.ndim != 2 or value.shape[0] != len(positions) or value.shape[1] <= 0:
                raise ValueError('Invalid native signal shape')
            if name == 'ids':
                if value.dtype not in (torch.int32, torch.int64):
                    raise ValueError('Native expert IDs must be integers')
            elif not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite native capture signal')
            if name not in self.arrays:
                shape = (len(self.layers), self.tokens, value.shape[1])
                size = len(self.layers) * self.tokens * (value.shape[1] * value.element_size() + 1)
                if self.allocated_bytes + size > self.max_bytes:
                    raise ValueError('Native capture buffer memory cap exceeded before allocation')
                self.arrays[name] = torch.empty(shape, dtype=value.dtype, device='cpu')
                self.coverage[name] = torch.zeros(shape[:2], dtype=torch.bool, device='cpu')
                self.allocated_bytes += size
            array, seen = self.arrays[name], self.coverage[name]
            if array.shape[2] != value.shape[1] or array.dtype != value.dtype:
                raise ValueError('Native signal dtype or width changed')
            previous = seen[index, positions]
            if bool(previous.any()) and not torch.equal(array[index, positions[previous]], value[previous]):
                raise RuntimeError('Recomputation changed a recorded native signal')
            array[index, positions] = value
            seen[index, positions] = True

    def require_complete(self):
        if set(self.arrays) != self.expected or any(not bool(x.all()) for x in self.coverage.values()):
            raise RuntimeError('Incomplete exact completion coverage or missing native capture signal')
        return self.arrays

    def for_storage(self):
        import torch
        arrays = self.require_complete()
        result = dict(arrays)
        weights = arrays['weights']
        if bool((weights < 0).any()) or bool((weights > torch.finfo(torch.float16).max).any()):
            raise ValueError('Native weights cannot be stored as finite nonnegative fp16')
        # Preserve rank hashes on arrays, then return an explicit storage rounding.
        result['weights'] = weights.to(torch.float16)
        return result


def verified_modular_backend(quant):
    """Only backends whose native select→forward_modular boundary is audited."""
    if getattr(quant, 'is_monolithic', True):
        raise RuntimeError('Unsupported monolithic dispatch; use an audited modular backend')
    allowed = {
        'fp8_backend': {'MARLIN', 'TRITON'},
        'nvfp4_backend': {'MARLIN'},
        'wna16_backend': {'MARLIN', 'TRITON'},
        'wna16_moe_backend': {'MARLIN', 'TRITON'},  # AutoGPTQMoEMethod (gptq_marlin alias)
        'mxfp4_backend': {'MARLIN', 'TRITON', 'TRITON_UNFUSED'},
        'unquantized_backend': {'TRITON'},
    }
    found = []
    for attribute, names in allowed.items():
        backend = getattr(quant, attribute, None)
        if backend is not None:
            name = getattr(backend, 'name', None)
            if name not in names:
                raise RuntimeError(f'Unverified native MoE backend: {attribute}={name}')
            found.append((attribute, name))
    if len(found) != 1:
        raise RuntimeError('Require one verified native modular MoE backend')
    return '/'.join(found[0])
