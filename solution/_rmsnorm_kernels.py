"""Original row-wise RMSNorm kernels for frozen fp32 Qwen2 norm weights.

One program handles one token. Reduction and arithmetic use fp32; each launch
combines the normalization's element-wise operations and row reduction.
"""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_forward_kernel(
    x_ptr, weight_ptr, out_ptr, inv_rms_ptr,
    WIDTH: tl.constexpr, EPSILON: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    valid = columns < WIDTH
    values = tl.load(x_ptr + row * WIDTH + columns, mask=valid, other=0.0)
    weights = tl.load(weight_ptr + columns, mask=valid, other=0.0)
    mean_square = tl.sum(values * values, axis=0) / WIDTH
    inv_rms = tl.rsqrt(mean_square + EPSILON)
    normalized = values * inv_rms
    tl.store(out_ptr + row * WIDTH + columns, normalized * weights, mask=valid)
    tl.store(inv_rms_ptr + row, inv_rms)


@triton.jit
def rmsnorm_input_grad_kernel(
    grad_ptr, x_ptr, weight_ptr, inv_rms_ptr, dx_ptr,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    valid = columns < WIDTH
    values = tl.load(x_ptr + row * WIDTH + columns, mask=valid, other=0.0)
    upstream = tl.load(grad_ptr + row * WIDTH + columns, mask=valid, other=0.0)
    weights = tl.load(weight_ptr + columns, mask=valid, other=0.0)
    inv_rms = tl.load(inv_rms_ptr + row)
    scaled_grad = upstream * weights
    projection = tl.sum(scaled_grad * values, axis=0) / WIDTH
    correction = values * (inv_rms * inv_rms * projection)
    input_grad = inv_rms * (scaled_grad - correction)
    tl.store(dx_ptr + row * WIDTH + columns, input_grad, mask=valid)
