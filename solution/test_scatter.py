"""Saved-activation and derivative contracts for the shared padding scatter."""

import gc
import weakref

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scatter_preserves_values_and_backward_without_saving_compact_rows(device, dtype):
    from solution.scatter import ScatterRows

    torch.manual_seed(321)
    source = torch.randn(5, 1536, device=device, dtype=dtype, requires_grad=True)
    reference_source = source.detach().clone().requires_grad_(True)
    compact = source * 1.25
    reference_compact = reference_source * 1.25
    indices = torch.tensor([7, 0, 11, 2, 5], device=device, dtype=torch.long)
    expected = reference_compact.new_zeros((12, 1536))
    expected.index_copy_(0, indices, reference_compact)
    compact_reference = weakref.ref(compact)
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype, tensor.data_ptr()))
        return tensor

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = ScatterRows.apply(compact, indices, 12)
    assert saved == [((5,), torch.int64, indices.data_ptr())]
    del compact
    gc.collect()
    assert compact_reference() is None, "Projection output values must not survive in the scatter graph"
    assert actual.dtype == dtype and actual.shape == (12, 1536)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())

    # Exercise noncontiguous upstream storage. Infinite gradients on omitted
    # padded rows must never enter the compact gradient or turn it into NaNs.
    upstream = torch.randn(12, 3072, device=device, dtype=dtype)[:, ::2]
    selected = torch.zeros(12, device=device, dtype=torch.bool)
    selected[indices] = True
    upstream[~selected] = float("inf")
    assert not upstream.is_contiguous()
    actual_gradient = torch.autograd.grad(actual, source, upstream)[0]
    reference_gradient = torch.autograd.grad(expected, reference_source, upstream)[0]
    assert torch.isfinite(actual_gradient).all()
    assert actual_gradient.dtype == dtype
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)
