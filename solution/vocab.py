"""Tiled token cross-entropy and inference-only entropy, with PyTorch fallbacks.

The kernels use fp32 arithmetic and never overwrite logits. Training loss
normalization, causal alignment and metrics masks remain in the callers.
"""

import torch
from torch.autograd.function import once_differentiable

VOCAB_BLOCK = 4096


def supports_cross_entropy(logits, targets):
    return (
        isinstance(logits, torch.Tensor)
        and isinstance(targets, torch.Tensor)
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[0] > 0
        and 0 < logits.shape[1] <= 262144
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and targets.device == logits.device
        and targets.dtype == torch.int64
        and targets.shape == (logits.shape[0],)
    )


def supports_entropy(logits):
    return (
        isinstance(logits, torch.Tensor)
        and not torch.is_grad_enabled()
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim >= 1
        and logits.numel() > 0
        and 0 < logits.shape[-1] <= 262144
        and logits.is_contiguous()
    )


def _tile_statistics(logits, with_entropy=False):
    from ._vocab_kernels import vocab_tile_stats_kernel

    rows, vocab = logits.shape
    tiles = (vocab + VOCAB_BLOCK - 1) // VOCAB_BLOCK
    maxima = torch.empty((rows, tiles), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    # The pointer is unused in the CE specialization; no moment is computed.
    moments = torch.empty_like(maxima) if with_entropy else maxima
    vocab_tile_stats_kernel[(rows, tiles)](
        logits, maxima, sums, moments, logits.stride(0),
        VOCAB=vocab, TILES=tiles, BLOCK=VOCAB_BLOCK,
        WITH_ENTROPY=with_entropy, num_warps=4, enable_fp_fusion=False,
    )
    return maxima, sums, moments


class _TokenCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        from ._vocab_kernels import vocab_ce_reduce_kernel

        labels = targets.contiguous()
        rows, vocab = logits.shape
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
        row_maxima = torch.empty_like(losses)
        inverse_sums = torch.empty_like(losses)
        with torch.cuda.device(logits.device):
            valid = (labels == ignore_index) | ((labels >= 0) & (labels < vocab))
            torch._assert_async(valid.all(), "Cross-entropy target is out of bounds")
            maxima, sums, _ = _tile_statistics(logits)
            tiles = maxima.shape[1]
            vocab_ce_reduce_kernel[(rows,)](
                logits, labels, maxima, sums, losses, row_maxima, inverse_sums,
                logits.stride(0), VOCAB=vocab, TILES=tiles,
                IGNORE_INDEX=ignore_index, TILE_BLOCK=1 << (tiles - 1).bit_length(),
                num_warps=4, enable_fp_fusion=False,
            )
        ctx.save_for_backward(logits, labels, row_maxima, inverse_sums)
        ctx.ignore_index = ignore_index
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_losses):
        from ._vocab_kernels import vocab_ce_backward_kernel

        logits, labels, row_maxima, inverse_sums = ctx.saved_tensors
        rows, vocab = logits.shape
        gradients = torch.empty(logits.shape, device=logits.device, dtype=torch.float32)
        upstream = grad_losses.contiguous()
        with torch.cuda.device(logits.device):
            vocab_ce_backward_kernel[(rows, (vocab + VOCAB_BLOCK - 1) // VOCAB_BLOCK)](
                logits, labels, row_maxima, inverse_sums, upstream, gradients,
                logits.stride(0), VOCAB=vocab, IGNORE_INDEX=ctx.ignore_index,
                BLOCK=VOCAB_BLOCK, num_warps=4, enable_fp_fusion=False,
            )
        return gradients, None, None


def per_token_cross_entropy(logits, targets, ignore_index=-100):
    """Cross-entropy with reduction='none'; first-order gradients on the fast path."""
    if not supports_cross_entropy(logits, targets) or not isinstance(ignore_index, int):
        return torch.nn.functional.cross_entropy(
            logits, targets, reduction="none", ignore_index=ignore_index,
        )
    return _TokenCrossEntropy.apply(logits, targets, ignore_index)


def entropy(logits):
    """Match TRL's per-token entropy; fuse the fp32 no-grad metric path only."""
    if not supports_entropy(logits):
        flat = logits.reshape(-1, logits.shape[-1])
        values = []
        for chunk in flat.split(128, dim=0):
            logps = torch.nn.functional.log_softmax(chunk, dim=-1)
            values.append(-(torch.exp(logps) * logps).sum(-1))
        return torch.cat(values, dim=0).reshape(logits.shape[:-1])

    from ._vocab_kernels import vocab_entropy_reduce_kernel

    flat = logits.view(-1, logits.shape[-1])
    output = torch.empty(flat.shape[0], device=logits.device, dtype=torch.float32)
    with torch.cuda.device(logits.device):
        maxima, sums, moments = _tile_statistics(flat, with_entropy=True)
        tiles = maxima.shape[1]
        vocab_entropy_reduce_kernel[(flat.shape[0],)](
            maxima, sums, moments, output, TILES=tiles,
            TILE_BLOCK=1 << (tiles - 1).bit_length(),
            num_warps=4, enable_fp_fusion=False,
        )
    return output.reshape(logits.shape[:-1])
