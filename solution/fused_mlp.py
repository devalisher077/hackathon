"""One autograd node for the task's frozen NF4 MLP, with exact dtype boundaries.

The three projection GEMMs and the organizer's SwiGLU kernels are retained.
Gate and up input gradients are rounded independently to fp16, then widened
and added in fp32. Dense weights are never retained across forward/backward.
Install this wrapper after the dtype patch and before the padding wrapper.
"""

from functools import lru_cache, wraps

import torch
from torch.autograd.function import once_differentiable


_MARKER = "_soup_fused_mlp_patch"


@lru_cache(maxsize=1)
def _backend():
    import bitsandbytes as bnb
    from bitsandbytes.backends.cuda import ops
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    return bnb, Qwen2MLP, ops


def _constant_tensor(value, device, dtype, *, elements=None):
    return (
        isinstance(value, torch.Tensor)
        and value.device == device
        and value.dtype == dtype
        and value.is_contiguous()
        and not value.requires_grad
        and (elements is None or value.numel() == elements)
    )


def _supported_projection(layer, x, linear_class):
    if type(layer) is not linear_class or "forward" in layer.__dict__ or hasattr(layer, "_hf_hook"):
        return False
    # Calling F.linear directly must not bypass an installed projection hook
    # or an unknown replacement of Linear4bit.forward.
    method = getattr(layer.forward, "__func__", None)
    if (
        method is not linear_class.forward
        or getattr(method, "__module__", None) != "bitsandbytes.nn.modules"
        or getattr(method, "__qualname__", None) != "Linear4bit.forward"
        or hasattr(method, "__wrapped__")
        or any(getattr(layer, name, None) for name in (
            "_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks"
        ))
        or layer.bias is not None
        or layer.compute_dtype != torch.float16
        or layer.compute_type_is_set is not True
        or any(parameter.requires_grad for parameter in layer.parameters())
    ):
        return False
    weight = layer.weight
    count = layer.in_features * layer.out_features
    state = getattr(weight, "quant_state", None)
    shape = getattr(state, "shape", None)
    if (
        not _constant_tensor(weight, x.device, torch.uint8, elements=(count + 1) // 2)
        or weight.ndim != 2
        or weight.shape[1] != 1
        or getattr(weight, "bnb_quantized", None) is not True
        or state is None
        or not isinstance(shape, (tuple, list, torch.Size))
        or tuple(shape) != (layer.out_features, layer.in_features)
        or getattr(state, "quant_type", None) != "nf4"
        or getattr(state, "dtype", None) != torch.float16
        or getattr(state, "blocksize", None) != 64
        or getattr(state, "nested", None) is not True
        or count % 64
    ):
        return False
    blocks = count // 64
    nested = getattr(state, "state2", None)
    return (
        nested is not None
        and getattr(nested, "blocksize", None) == 256
        and getattr(nested, "dtype", None) == torch.float32
        and getattr(nested, "nested", None) is False
        and _constant_tensor(getattr(state, "absmax", None), x.device, torch.uint8, elements=blocks)
        and _constant_tensor(getattr(nested, "absmax", None), x.device, torch.float32, elements=(blocks + 255) // 256)
        and _constant_tensor(getattr(nested, "code", None), x.device, torch.float32, elements=256)
        and _constant_tensor(getattr(state, "offset", None), x.device, torch.float32, elements=1)
    )


def supported_fused_mlp(module, x):
    """Only the known first-order, frozen, bias-free fp16 NF4 training path."""
    if (
        not isinstance(x, torch.Tensor)
        or not x.is_cuda
        or x.dtype != torch.float32
        or x.ndim not in (2, 3)
        or not x.is_contiguous()
        or x.numel() == 0
        or not x.requires_grad
        or not torch.is_grad_enabled()
        or not getattr(module, "training", False)
        or torch.__version__.split("+")[0] != "2.11.0"
    ):
        return False
    bnb, mlp_class, backend = _backend()
    if (
        bnb.__version__ != "0.50.2"
        or type(module) is not mlp_class
        or getattr(module.config, "hidden_act", None) != "silu"
        or x.shape[-1] != module.hidden_size
        or any(parameter.requires_grad for parameter in module.parameters())
    ):
        return False
    # Global hooks also observe child projections under the ordinary path.
    hook_module = torch.nn.modules.module
    if any(getattr(hook_module, name, None) for name in (
        "_global_forward_pre_hooks", "_global_forward_hooks", "_global_backward_pre_hooks", "_global_backward_hooks"
    )):
        return False
    gate, up, down = module.gate_proj, module.up_proj, module.down_proj
    if not (
        all(_supported_projection(layer, x, bnb.nn.Linear4bit) for layer in (gate, up, down))
        and gate.in_features == up.in_features == module.hidden_size
        and gate.out_features == up.out_features == down.in_features == module.intermediate_size
        and down.out_features == module.hidden_size
    ):
        return False
    # Small matrices can use bnb's separate fused NF4 GEMM. Restrict this
    # replacement to shapes for which the original uses dequant + F.linear.
    rows = x.numel() // x.shape[-1]
    selector = getattr(backend, "_gemm_4bit_use_custom_fn", None)
    ceiling = getattr(backend, "_gemm_4bit_custom_max_m", None)
    if not callable(selector) or type(ceiling) is not int:
        return False
    for layer in (gate, up, down):
        if rows <= ceiling and layer.in_features % 64 == 0:
            if selector(x.device.index, torch.float16, rows, layer.out_features, layer.in_features):
                return False
    return True


def _weight(layer):
    bnb, _, _ = _backend()
    # Resolve the attribute each call so the optional decoded-scale cache is
    # honored regardless of whether it was installed before or after us.
    return bnb.functional.dequantize_4bit(layer.weight, layer.weight.quant_state)


class _FrozenMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module):
        from .swiglu import example_swiglu_fwd_kernel

        shape = x.shape
        half_input = x.view(-1, shape[-1]).to(torch.float16)
        weight = _weight(module.gate_proj)
        gate = torch.nn.functional.linear(half_input, weight)
        del weight
        weight = _weight(module.up_proj)
        up = torch.nn.functional.linear(half_input, weight)
        del weight, half_input
        activation = torch.empty_like(gate)
        elements = gate.numel()
        # Reuse the organizer example unchanged; it is not a new own kernel.
        example_swiglu_fwd_kernel[((elements + 1023) // 1024,)](
            gate, up, activation, elements, BLOCK=1024
        )
        weight = _weight(module.down_proj)
        output = torch.nn.functional.linear(activation, weight).float()
        ctx.save_for_backward(gate, up)
        ctx.module = module
        ctx.input_shape = shape
        return output.view(shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        from .swiglu import example_swiglu_bwd_kernel
        from ._mlp_grad_kernels import mlp_input_grad_sum_kernel

        gate, up = ctx.saved_tensors
        # The pinned PyTorch method releases internal saved-tensor references
        # only when the graph is not retained. Local gate/up references stay
        # alive through SwiGLU backward and can then be released before GEMMs.
        ctx.maybe_clear_saved_tensors()
        module = ctx.module
        half_gradient = gradient.reshape(-1, gradient.shape[-1]).to(torch.float16)
        weight = _weight(module.down_proj)
        activation_gradient = torch.matmul(half_gradient, weight)
        del weight, half_gradient
        gate_gradient, up_gradient = torch.empty_like(gate), torch.empty_like(up)
        elements = gate.numel()
        example_swiglu_bwd_kernel[((elements + 1023) // 1024,)](
            activation_gradient, gate, up, gate_gradient, up_gradient, elements, BLOCK=1024
        )
        del gate, up, activation_gradient
        weight = _weight(module.gate_proj)
        gate_dx = torch.matmul(gate_gradient, weight)
        del weight, gate_gradient
        weight = _weight(module.up_proj)
        up_dx = torch.matmul(up_gradient, weight)
        del weight, up_gradient
        output = torch.empty(gate_dx.shape, device=gate_dx.device, dtype=torch.float32)
        elements = output.numel()
        mlp_input_grad_sum_kernel[((elements + 1023) // 1024,)](
            gate_dx, up_dx, output, elements,
            BLOCK=1024, num_warps=4, enable_fp_fusion=False,
        )
        return output.view(ctx.input_shape), None


def fused_mlp_forward(module, x):
    """Execute after supported_fused_mlp; preserve fp32 caller output and dx."""
    return _FrozenMLP.apply(x, module)


def patch_qwen2_fused_mlp():
    """Install before padding compaction; retain the existing fallback path."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward
    if getattr(original, _MARKER, False):
        return True
    if getattr(original, "_soup_padding_mlp_patch", False):
        # Wrapping outside padding would bypass the already-installed gather.
        return False

    @wraps(original)
    def forward(self, x):
        if not supported_fused_mlp(self, x):
            return original(self, x)
        return fused_mlp_forward(self, x)

    setattr(forward, _MARKER, True)
    modeling_qwen2.Qwen2MLP.forward = forward
    return True
