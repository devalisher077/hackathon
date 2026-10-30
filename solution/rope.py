"""Fuse the task's fp32 Qwen2 rotary operations without changing precision."""

from functools import wraps

import torch
from torch.autograd.function import once_differentiable


_MARKER = "_soup_rope_pair_patch"


def _supported(q, k, cos, sin, unsqueeze_dim):
    if unsqueeze_dim != 1 or not all(isinstance(x, torch.Tensor) for x in (q, k, cos, sin)):
        return False
    if not q.is_cuda or any(x.device != q.device or x.dtype != torch.float32 for x in (q, k, cos, sin)):
        return False
    if q.ndim != 4 or k.ndim != 4 or cos.ndim != 3 or sin.ndim != 3:
        return False
    batch, heads, time, width = q.shape
    if not (batch > 0 and time > 0 and heads > 0 and k.shape[1] > 0 and 0 < width <= 256 and width % 2 == 0):
        return False
    if k.shape[0] != batch or k.shape[2:] != (time, width) or (heads + k.shape[1]) * width > 8192:
        return False
    # Qwen2's projection outputs are contiguous B,T,H,D, viewed as B,H,T,D.
    for x in (q, k):
        if x.stride(3) != 1 or x.stride(1) != width or x.stride(2) != x.shape[1] * width:
            return False
        if x.stride(0) != time * x.shape[1] * width:
            return False
    for x in (cos, sin):
        if x.requires_grad or x.shape[0] not in (1, batch) or x.shape[1:] != (time, width) or x.stride(2) != 1:
            return False
    return True


def _empty(batch, heads, time, width, device):
    return torch.empty((batch, time, heads, width), device=device, dtype=torch.float32).transpose(1, 2)


class _RotaryPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        from ._rope_kernels import rope_pair_forward_kernel

        batch, heads, time, width = q.shape
        kheads = k.shape[1]
        oq, ok = _empty(batch, heads, time, width, q.device), _empty(batch, kheads, time, width, q.device)
        block = 1 << ((heads + kheads) * width - 1).bit_length()
        rope_pair_forward_kernel[(batch * time,)](
            q, k, cos, sin, oq, ok, q.stride(0), q.stride(2), k.stride(0), k.stride(2),
            cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
            T=time, D=width, Q=heads * width, K=kheads * width,
            COS_BROADCAST=cos.shape[0] == 1, SIN_BROADCAST=sin.shape[0] == 1,
            BLOCK=block, num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(cos, sin)
        ctx.shape = batch, heads, kheads, time, width, block
        ctx.set_materialize_grads(False)
        return oq, ok

    @staticmethod
    @once_differentiable
    def backward(ctx, gq, gk):
        if gq is None and gk is None:
            return None, None, None, None
        from ._rope_kernels import rope_pair_backward_kernel

        cos, sin = ctx.saved_tensors
        batch, heads, kheads, time, width, block = ctx.shape
        missing_q, missing_k = gq is None, gk is None
        if missing_q:
            gq = torch.zeros((batch, heads, time, width), device=cos.device, dtype=torch.float32)
        if missing_k:
            gk = torch.zeros((batch, kheads, time, width), device=cos.device, dtype=torch.float32)
        dq, dk = _empty(batch, heads, time, width, cos.device), _empty(batch, kheads, time, width, cos.device)
        rope_pair_backward_kernel[(batch * time,)](
            gq, gk, cos, sin, dq, dk, *gq.stride(), *gk.stride(),
            cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
            T=time, D=width, Q=heads * width, K=kheads * width,
            COS_BROADCAST=cos.shape[0] == 1, SIN_BROADCAST=sin.shape[0] == 1,
            BLOCK=block, num_warps=4, enable_fp_fusion=False,
        )
        return None if missing_q else dq, None if missing_k else dk, None, None


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def rotary(q, k, cos, sin, unsqueeze_dim=1):
        if not _supported(q, k, cos, sin, unsqueeze_dim):
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return _RotaryPair.apply(q, k, cos, sin)

    setattr(rotary, _MARKER, True)
    modeling_qwen2.apply_rotary_pos_emb = rotary
