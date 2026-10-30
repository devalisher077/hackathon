"""Padding scatter whose backward retains row indices, not activation values."""

import torch


class ScatterRows(torch.autograd.Function):
    """Place unique compact rows in a zero-filled matrix, preserving dtype.

    PyTorch 2.11's ordinary index_copy_ autograd saves its source tensor.
    These tokenwise projection outputs need not survive the forward pass:
    their derivative is just the upstream gradient gathered at the same rows.
    Callers supply a two-dimensional source and unique in-bounds row indices.
    """

    @staticmethod
    def forward(ctx, compact, indices, full_rows):
        output = compact.new_zeros((full_rows, compact.shape[1]))
        output.index_copy_(0, indices, compact)
        ctx.save_for_backward(indices)
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:
            return None, None, None
        (indices,) = ctx.saved_tensors
        return gradient.index_select(0, indices), None, None
