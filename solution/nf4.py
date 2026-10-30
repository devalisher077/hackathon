"""Replace only bitsandbytes 0.50.2's CUDA NF4 unpack/scale helper.

Nested (double-quantized) scales are still reconstructed by bitsandbytes;
its matmul, autograd, dtype choices and packed-weight transpose convention
are untouched. Unsupported inputs call the original backend helper.

The numeric table below is NF4 format data, from bitsandbytes' documented
``get_4bit_type('nf4')`` in functional.py (0.50.2), not a copied kernel:
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py
The Triton unpack/scale implementation is written for this solution.
"""

from functools import wraps
from importlib import import_module

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
