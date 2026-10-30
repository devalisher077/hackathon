"""Keep the frozen NF4 MLP's already-fp16 intermediates in their compute dtype.

The organizer's SwiGLU implementation is reused unchanged and is not claimed as
an original kernel. In the supported path, bitsandbytes computes each projection
in fp16 but widens its result to its input's fp32 dtype. Removing the gate/up
widening avoids storing those two large activations in fp32. The caller still
receives the same fp32 MLP result.
"""

from functools import wraps

import torch


def _supported_projection(layer, x, linear_class):
    if type(layer) is not linear_class:
        # In particular, do not bypass a PEFT wrapper or a custom subclass.
        return False
    weight = layer.weight
    state = getattr(weight, "quant_state", None)
    return (
        isinstance(weight, torch.Tensor)
        and not weight.requires_grad
        and weight.is_cuda
        and weight.device == x.device
        and weight.dtype == torch.uint8
        and layer.bias is None
        and layer.compute_dtype == torch.float16
        and layer.compute_type_is_set is True
        and state is not None
        and getattr(state, "quant_type", None) == "nf4"
        and tuple(state.shape) == (layer.out_features, layer.in_features)
    )


def supported_mlp(module, x):
    """Restrict the optimization to the task's plain, frozen fp16 NF4 MLP."""
    if not (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.float32
        and x.ndim in (2, 3)
        and x.numel() > 0
        and x.is_contiguous()
    ):
        return False

    # GPU dependencies stay lazy so CPU and mocked-framework calls preserve
    # their original path without importing bitsandbytes or Triton.
    from bitsandbytes.nn import Linear4bit
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    if type(module) is not Qwen2MLP:
        return False
    if getattr(module.config, "hidden_act", None) != "silu":
        return False
    if x.shape[-1] != module.hidden_size:
        return False

    gate, up, down = module.gate_proj, module.up_proj, module.down_proj
    if not all(_supported_projection(layer, x, Linear4bit) for layer in (gate, up, down)):
        return False
    return (
        gate.in_features == up.in_features == x.shape[-1]
        and gate.out_features == up.out_features == down.in_features
        and down.out_features == x.shape[-1]
    )


def mlp_forward(module, x):
    """Compute the guarded path using the unchanged organizer SwiGLU kernels."""
    from .swiglu import swiglu

    # These MUST be two distinct casts. Reusing a shared half input would sum
    # the two projection gradients in fp16 before casting back to fp32, whereas
    # the original independent Linear4bit calls accumulate them in fp32.
    gate = module.gate_proj(x.to(torch.float16))
    up = module.up_proj(x.to(torch.float16))

    # The original down projection casts the fp32 SwiGLU result to fp16 before
    # its matmul. Storing SwiGLU directly in fp16 keeps that rounding boundary.
    # Its backward already receives a gradient originating in fp16, and each
    # projection needs an fp16 output gradient, so the two saved half-valued
    # projection results suffice without changing their represented values.
    output = module.down_proj(swiglu(gate, up))
    return output.to(x.dtype)


def patch_qwen2_mlp_dtype():
    """Wrap the currently installed MLP path once, retaining its fallbacks."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward
    if getattr(original, "_soup_mlp_dtype_patch", False):
        return

    @wraps(original)
    def forward(self, x):
        if not supported_mlp(self, x):
            return original(self, x)
        return mlp_forward(self, x)

    forward._soup_mlp_dtype_patch = True
    modeling_qwen2.Qwen2MLP.forward = forward
