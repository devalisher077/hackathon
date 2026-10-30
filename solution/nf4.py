"""Replace only bitsandbytes 0.50.2's CUDA NF4 unpack/scale helper.

Nested (double-quantized) scales are still reconstructed by bitsandbytes;
its matmul, autograd, dtype choices and packed-weight transpose convention
are untouched. Unsupported inputs call the original backend helper.

The numeric table below is NF4 format data, from bitsandbytes' documented
``get_4bit_type('nf4')`` in functional.py (0.50.2), not a copied kernel:
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py
The Triton unpack/scale implementation is written for this solution.
"""

from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
import weakref

import torch


_NF4_LEVELS = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
)
_BLOCKS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
_LEVELS_BY_DEVICE = {}
_SCALE_CACHE = OrderedDict()
_SCALE_CACHE_BYTES = 0
_SCALE_CACHE_LIMIT = 128 * 2**20
_SCALE_MARKER = "_soup_nf4_scale_cache"


@dataclass
class _ScaleEntry:
    references: tuple
    signature: tuple
    decoded: torch.Tensor


def _remove_scale_entry(key):
    global _SCALE_CACHE_BYTES
    entry = _SCALE_CACHE.pop(key, None)
    if entry is not None:
        _SCALE_CACHE_BYTES -= entry.decoded.numel() * entry.decoded.element_size()


def _clear_scale_cache():
    global _SCALE_CACHE_BYTES
    _SCALE_CACHE.clear()
    _SCALE_CACHE_BYTES = 0


def _tensor_signature(tensor):
    # Version counters catch ordinary in-place updates. A replaced .data
    # storage is caught by its pointer. Frozen inputs are never modified here.
    return (tensor.data_ptr(), tensor._version, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device)


def _scale_inputs_supported(quantized, absmax, code, offset):
    values = quantized, absmax, code, offset
    return (
        all(isinstance(value, torch.Tensor) for value in values)
        and quantized.is_cuda
        and all(value.device == quantized.device and value.is_contiguous()
                and not value.requires_grad for value in values)
        and quantized.dtype == torch.uint8
        and quantized.numel() > 0
        and all(value.dtype == torch.float32 for value in (absmax, code, offset))
        and absmax.numel() >= (quantized.numel() + 255) // 256
        and code.numel() == 256
        and offset.numel() == 1
        and quantized.numel() * 4 <= _SCALE_CACHE_LIMIT
    )


def _cached_scales(backend, quantized, absmax, code, offset):
    """Reuse exactly bnb's decoded nested scales, including its fp32 offset.

The same inputs reach the CUDA forward fallback and functional backward
dequantizer, so both share this entry. Stream-local entries preserve CUDA
ordering without introducing synchronization or cross-stream dependencies.
"""
    global _SCALE_CACHE_BYTES
    if not _scale_inputs_supported(quantized, absmax, code, offset):
        return None
    values = quantized, absmax, code, offset
    try:
        signature = tuple(_tensor_signature(value) for value in values)
    except RuntimeError:
        # Inference tensors may not expose version counters; remain uncached.
        return None
    stream = torch.cuda.current_stream(quantized.device).cuda_stream
    key = (*map(id, values), stream)
    entry = _SCALE_CACHE.get(key)
    if (entry is not None and entry.signature == signature
            and all(reference() is value for reference, value in zip(entry.references, values))):
        _SCALE_CACHE.move_to_end(key)
        return entry.decoded
    _remove_scale_entry(key)
    with torch.no_grad():
        decoded = torch.empty_like(quantized, dtype=torch.float32)
        backend._dequantize_blockwise_impl(quantized, absmax, code, 256, torch.float32, out=decoded)
        decoded.add_(offset)
    size = decoded.numel() * decoded.element_size()
    while _SCALE_CACHE and _SCALE_CACHE_BYTES + size > _SCALE_CACHE_LIMIT:
        _remove_scale_entry(next(iter(_SCALE_CACHE)))

    def release(reference):
        current = _SCALE_CACHE.get(key)
        if current is not None and any(saved is reference for saved in current.references):
            _remove_scale_entry(key)

    references = tuple(weakref.ref(value, release) for value in values)
    _SCALE_CACHE[key] = _ScaleEntry(references, signature, decoded)
    _SCALE_CACHE_BYTES += size
    return decoded


def _supported(A, absmax, blocksize, quant_type, dtype, out):
    if not all(isinstance(value, torch.Tensor) for value in (A, absmax, out)):
        return False
    return (
        quant_type == "nf4"
        and blocksize in _BLOCKS
        and A.is_cuda
        and A.dtype == torch.uint8
        and A.is_contiguous()
        and absmax.device == A.device
        and absmax.dtype == torch.float32
        and absmax.is_contiguous()
        and out.device == A.device
        and dtype in (torch.float16, torch.float32)
        and out.dtype == dtype
        and out.is_contiguous()
        and not out.requires_grad
        and out.numel() > 0
        and A.numel() * 2 >= out.numel()
        and absmax.numel() >= (out.numel() + blocksize - 1) // blocksize
    )


def _launch_nf4(A, absmax, blocksize, out):
    # Importing this module on a CPU machine never requires Triton or bnb.
    from ._nf4_kernels import nf4_unpack_scale_kernel

    elements = out.numel()
    tile = 1024
    with torch.cuda.device(A.device):
        levels = _LEVELS_BY_DEVICE.get(A.device)
        if levels is None:
            levels = torch.tensor(_NF4_LEVELS, device=A.device, dtype=torch.float32)
            _LEVELS_BY_DEVICE[A.device] = levels
        nf4_unpack_scale_kernel[((elements + tile - 1) // tile,)](
            A, absmax, levels, out, elements,
            QUANT_BLOCK=blocksize, TILE=tile,
            num_warps=4, enable_fp_fusion=False,
        )


def patch_nf4():
    """Idempotently patch the pinned bnb CUDA backend's shared dequant helper."""
    backend = import_module("bitsandbytes.backends.cuda.ops")
    original = backend._dequantize_4bit_impl
    if getattr(original, "_soup_custom_nf4", False):
        return

    @wraps(original)
    def dequantize(A, absmax, blocksize, quant_type, dtype, out):
        if not _supported(A, absmax, blocksize, quant_type, dtype, out):
            return original(A, absmax, blocksize, quant_type, dtype, out=out)
        _launch_nf4(A, absmax, blocksize, out)
        # The backend helper writes the caller's buffer and returns None.
        return None

    dequantize._soup_custom_nf4 = True
    backend._dequantize_4bit_impl = dequantize


def patch_nf4_scale_cache():
    """Optionally cache frozen double-quantized scales in bnb's two callers.

Dense NF4 weights are still unpacked for every forward/backward operation,
through the original helper (and therefore our kernel when installed). Neither
the quantization state nor any original parameter/tensor is overwritten.
"""
    bnb = import_module("bitsandbytes")
    if getattr(bnb, "__version__", None) != "0.50.2":
        return False
    backend = import_module("bitsandbytes.backends.cuda.ops")
    functional = import_module("bitsandbytes.functional")
    original_forward = backend._dequant_linear_fallback
    if not getattr(original_forward, _SCALE_MARKER, False):
        @wraps(original_forward)
        def forward(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                    absmax_8bit=None, absmax_code=None, absmax_offset=None):
            decoded = None
            if (quant_type == "nf4" and isinstance(B, torch.Tensor) and B.is_cuda
                    and B.dtype == torch.uint8 and not B.requires_grad
                    and isinstance(A, torch.Tensor) and A.device == B.device
                    and A.dtype in (torch.float16, torch.float32)
                    and isinstance(absmax_8bit, torch.Tensor) and absmax_8bit.device == B.device):
                decoded = _cached_scales(backend, absmax_8bit, absmax, absmax_code, absmax_offset)
            if decoded is not None:
                return original_forward(A, B, shapeB, decoded, blocksize, quant_type, bias=bias)
            return original_forward(A, B, shapeB, absmax, blocksize, quant_type, bias=bias,
                                    absmax_8bit=absmax_8bit, absmax_code=absmax_code,
                                    absmax_offset=absmax_offset)

        setattr(forward, _SCALE_MARKER, True)
        backend._dequant_linear_fallback = forward

    original_dequant = functional.dequantize_4bit
    if not getattr(original_dequant, _SCALE_MARKER, False):
        @wraps(original_dequant)
        def dequantize(A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4"):
            state2 = getattr(quant_state, "state2", None)
            decoded = None
            if (isinstance(A, torch.Tensor) and A.is_cuda and A.dtype == torch.uint8
                    and not A.requires_grad and getattr(quant_state, "nested", False)
                    and getattr(quant_state, "quant_type", None) == "nf4"
                    and getattr(quant_state, "dtype", None) in (torch.float16, torch.float32)
                    and isinstance(getattr(quant_state, "absmax", None), torch.Tensor)
                    and quant_state.absmax.device == A.device
                    and getattr(state2, "blocksize", None) == 256
                    and getattr(state2, "dtype", None) == torch.float32
                    and not getattr(state2, "nested", True)):
                decoded = _cached_scales(backend, quant_state.absmax, state2.absmax,
                                         state2.code, quant_state.offset)
            if decoded is not None:
                # Original metadata is read anew on every call; this shallow
                # view changes only how the already-decoded scales are supplied.
                quant_state = copy(quant_state)
                quant_state.absmax = decoded
                quant_state.nested = False
                quant_state.state2 = None
                quant_state.offset = None
            return original_dequant(A, quant_state, absmax, out, blocksize, quant_type)

        setattr(dequantize, _SCALE_MARKER, True)
        functional.dequantize_4bit = dequantize
    return True
