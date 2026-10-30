"""Original NF4 unpack/scale kernel; the caller supplies decoded fp32 scales.

The format has two indices per byte, high nibble first, with one scale for
each quantization block. This kernel does not quantize or change weights.
"""

import triton
import triton.language as tl


@triton.jit
def nf4_unpack_scale_kernel(
    packed_ptr, absmax_ptr, levels_ptr, output_ptr, elements,
    QUANT_BLOCK: tl.constexpr, TILE: tl.constexpr,
):
    element = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = element < elements
    packed = tl.load(packed_ptr + element // 2, mask=valid, other=0).to(tl.int32)
    shift = tl.where(element % 2 == 0, 4, 0)
    index = (packed >> shift) & 15
    level = tl.load(levels_ptr + index)
    scale = tl.load(absmax_ptr + element // QUANT_BLOCK, mask=valid, other=0.0)
    value = level * scale
    tl.store(output_ptr + element, value.to(output_ptr.dtype.element_ty), mask=valid)
