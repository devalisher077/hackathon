"""Fused RMSNorm for the fp32, frozen-weight path used by this LoRA task.

The mathematical reference is Qwen2RMSNorm in Transformers 5.16.1. The Triton
kernels are written for this solution; no third-party kernel code is used.
Other dtypes, trainable weights and CPU calls preserve the PyTorch operation.
"""

from functools import wraps

import torch
from torch.autograd.function import once_differentiable


def _supported(x, weight):
    return (
        isinstance(x, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and x.is_cuda
        and weight.device == x.device
        and x.dtype == torch.float32
        and weight.dtype == torch.float32
        and not weight.requires_grad
        and x.ndim >= 1
        and 0 < x.shape[-1] <= 8192
        and x.numel() > 0
        and weight.shape == (x.shape[-1],)
    )


def _reference(x, weight, eps):
    values = x.float()
    variance = values.pow(2).mean(-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + eps)
    return weight * normalized.to(x.dtype)


class _FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        from ._rmsnorm_kernels import rmsnorm_forward_kernel

        values = x.contiguous()
        weights = weight.contiguous()
        width = values.shape[-1]
        rows = values.numel() // width
        block = 1 << (width - 1).bit_length()
        output = torch.empty_like(values)
        inverse_rms = torch.empty(rows, device=values.device, dtype=torch.float32)
        # Avoid fusing multiply/add across the reference's fp32 operation steps.
        with torch.cuda.device(values.device):
            rmsnorm_forward_kernel[(rows,)](
                values, weights, output, inverse_rms,
                WIDTH=width, EPSILON=float(eps), BLOCK=block,
                num_warps=4, enable_fp_fusion=False,
            )
        ctx.save_for_backward(values, weights, inverse_rms)
        ctx.width = width
        ctx.block = block
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        from ._rmsnorm_kernels import rmsnorm_input_grad_kernel

        values, weights, inverse_rms = ctx.saved_tensors
        upstream = grad_output.contiguous()
        input_grad = torch.empty_like(values)
        with torch.cuda.device(values.device):
            rmsnorm_input_grad_kernel[(inverse_rms.numel(),)](
                upstream, values, weights, inverse_rms, input_grad,
                WIDTH=ctx.width, BLOCK=ctx.block,
                num_warps=4, enable_fp_fusion=False,
            )
        return input_grad, None, None


def rmsnorm(x, weight, eps=1e-6):
    """Preserve Qwen2's dtype rules; fuse only frozen-weight CUDA fp32 inputs."""
    if not _supported(x, weight):
        return _reference(x, weight, eps)
    return _FrozenRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Patch only Qwen2 RMSNorm; repeat calls do not nest wrappers."""
    from transformers.models.qwen2 import modeling_qwen2

    norm_class = modeling_qwen2.Qwen2RMSNorm
    original = norm_class.forward
    if getattr(original, "_soup_custom_rmsnorm", False):
        return

    @wraps(original)
    def forward(self, hidden_states):
        if not _supported(hidden_states, self.weight):
            return original(self, hidden_states)
        return _FrozenRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)

    forward._soup_custom_rmsnorm = True
    norm_class.forward = forward
