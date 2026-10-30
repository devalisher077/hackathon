"""Original fp32 accumulation of the MLP's independently rounded branches."""

import triton
import triton.language as tl


@triton.jit
def mlp_input_grad_sum_kernel(gate_gradient, up_gradient, output, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    gate = tl.load(gate_gradient + offsets, mask=valid, other=0.0).to(tl.float32)
    up = tl.load(up_gradient + offsets, mask=valid, other=0.0).to(tl.float32)
    tl.store(output + offsets, gate + up, mask=valid)
