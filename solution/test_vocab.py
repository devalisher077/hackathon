"""Numerical checks for the vocabulary reductions, including their training gradients.

Run ``python -m pytest solution/test_vocab.py -v`` on the Kaggle T4. CPU
fallback checks do not validate Triton; all CUDA cases must also pass.
"""

import pytest
import torch
import torch.nn.functional as F


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# Keep the reference independent of any patch applied after this module loads.
_REFERENCE_CROSS_ENTROPY = F.cross_entropy
while hasattr(_REFERENCE_CROSS_ENTROPY, "__wrapped__"):
    _REFERENCE_CROSS_ENTROPY = _REFERENCE_CROSS_ENTROPY.__wrapped__


def _reference_entropy(logits):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def _targets(rows, width, device, ignore_index=-100):
    target = (torch.arange(rows, device=device, dtype=torch.long) * 149 + 3) % width
    if rows > 1:
        target[1] = ignore_index
    return target


def _check_cross_entropy(logits, targets, ignore_index=-100, fast=False):
    from solution.vocab import per_token_cross_entropy, supports_cross_entropy

    reference_logits = logits.detach().clone().requires_grad_(True)
    if fast:
        assert supports_cross_entropy(logits, targets)
    expected = _REFERENCE_CROSS_ENTROPY(
        reference_logits, targets, ignore_index=ignore_index, reduction="none"
    )
    actual = per_token_cross_entropy(logits, targets, ignore_index=ignore_index)
    assert actual.shape == targets.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)

    # Negative and non-unit weights catch omitted upstream scaling and any
    # accidental averaging inside the per-token primitive.
    upstream = torch.linspace(-0.7, 1.3, targets.numel(), device=logits.device,
                              dtype=actual.dtype).reshape_as(actual)
    dx = torch.autograd.grad(actual, logits, grad_outputs=upstream)[0]
    expected_dx = torch.autograd.grad(expected, reference_logits, grad_outputs=upstream)[0]
    torch.testing.assert_close(dx, expected_dx, rtol=5e-4, atol=2e-6)
    assert torch.equal(dx[targets == ignore_index], torch.zeros_like(dx[targets == ignore_index]))


@CUDA
@pytest.mark.parametrize("shape", [(7, 4097), (3, 151936), (4 * 17, 4097), (3, 8192)])
def test_cuda_cross_entropy_values_and_gradient(shape):
    torch.manual_seed(101)
    logits = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    _check_cross_entropy(logits, _targets(*shape, "cuda"), fast=True)


@CUDA
@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_cuda_cross_entropy_custom_ignore_index(ignore_index):
    torch.manual_seed(102)
    logits = torch.randn(5, 4097, device="cuda", requires_grad=True)
    target = _targets(5, 4097, "cuda", ignore_index=ignore_index)
    target[-1] = ignore_index
    _check_cross_entropy(logits, target, ignore_index=ignore_index, fast=True)


@CUDA
@pytest.mark.parametrize("normalization", ["infinite_upstream", "zero_valid_count"])
def test_cuda_all_ignored_gradients_stay_zero(normalization):
    from solution.vocab import per_token_cross_entropy

    logits = torch.randn(3, 4097, device="cuda", requires_grad=True)
    targets = torch.full((3,), -100, device="cuda", dtype=torch.long)
    loss = per_token_cross_entropy(logits, targets)
    assert torch.equal(loss, torch.zeros_like(loss))
    if normalization == "infinite_upstream":
        dx = torch.autograd.grad(loss, logits, torch.full_like(loss, float("inf")))[0]
    else:
        # This normalization produces NaN loss and infinite upstream gradients.
        # Ignored rows still have a zero derivative, as in PyTorch's NLL loss.
        mean = loss.sum() / (targets != -100).sum()
        assert torch.isnan(mean)
        dx = torch.autograd.grad(mean, logits)[0]
    assert torch.equal(dx, torch.zeros_like(dx))


@CUDA
@pytest.mark.parametrize("offset, scale", [(0.0, 0.0), (1e6, 1.0), (-1e6, 1.0)])
def test_cuda_cross_entropy_stable_for_zero_and_large_offsets(offset, scale):
    torch.manual_seed(103)
    logits = (torch.randn(4, 4097, device="cuda") * scale + offset).requires_grad_(True)
    _check_cross_entropy(logits, _targets(4, 4097, "cuda"), fast=True)


@CUDA
def test_cuda_cross_entropy_strided_rows_and_targets():
    torch.manual_seed(104)
    logits = torch.randn(12, 4097, device="cuda")[::2].detach().requires_grad_(True)
    target_storage = torch.zeros(12, device="cuda", dtype=torch.long)
    targets = target_storage[::2]
    targets.copy_(_targets(6, 4097, "cuda"))
    assert logits.stride(0) == 2 * logits.shape[1] and logits.stride(1) == 1
    assert not targets.is_contiguous()
    _check_cross_entropy(logits, targets, fast=True)


@CUDA
@pytest.mark.parametrize("denominator", ["valid_tokens", "external_scalar"])
def test_cuda_cross_entropy_preserves_external_loss_normalization(denominator):
    from solution.vocab import per_token_cross_entropy

    torch.manual_seed(105)
    logits = torch.randn(12, 4097, device="cuda", requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    targets = _targets(12, 4097, "cuda")
    targets[4:7] = -100
    divisor = (targets != -100).sum() if denominator == "valid_tokens" else 37.5
    actual = per_token_cross_entropy(logits, targets).sum() / divisor
    expected = _REFERENCE_CROSS_ENTROPY(ref_logits, targets, reduction="sum") / divisor
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    scale = 3.25
    dx = torch.autograd.grad(actual * scale, logits)[0]
    ref_dx = torch.autograd.grad(expected * scale, ref_logits)[0]
    torch.testing.assert_close(dx, ref_dx, rtol=5e-4, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cpu_cross_entropy_fallback(dtype):
    from solution.vocab import supports_cross_entropy

    torch.manual_seed(106)
    logits = torch.randn(5, 17, dtype=dtype, requires_grad=True)
    targets = _targets(5, 17, "cpu")
    assert not supports_cross_entropy(logits, targets)
    _check_cross_entropy(logits, targets)


@CUDA
def test_cuda_fp16_cross_entropy_fallback():
    from solution.vocab import supports_cross_entropy

    torch.manual_seed(107)
    logits = torch.randn(5, 4097, device="cuda", dtype=torch.float16, requires_grad=True)
    targets = _targets(5, 4097, "cuda")
    assert not supports_cross_entropy(logits, targets)
    _check_cross_entropy(logits, targets)


def test_cpu_invalid_target_raises_instead_of_silently_masking():
    from solution.vocab import per_token_cross_entropy

    logits = torch.randn(2, 17, requires_grad=True)
    targets = torch.tensor([0, 17], dtype=torch.long)
    with pytest.raises((IndexError, RuntimeError), match="(?i)(target|bound|range)"):
        per_token_cross_entropy(logits, targets)


@CUDA
@pytest.mark.parametrize("shape", [(7, 4097), (2, 3, 151936), (4, 17, 4097)])
def test_cuda_entropy_no_grad_preserves_token_shape(shape):
    from solution.vocab import entropy, supports_entropy

    torch.manual_seed(108)
    # A requires-grad input is common during training; the surrounding no_grad
    # context, rather than the input flag, controls this metric's fast path.
    logits = torch.randn(shape, device="cuda", requires_grad=True)
    with torch.no_grad():
        assert supports_entropy(logits)
        actual = entropy(logits)
        expected = _reference_entropy(logits)
    assert actual.shape == logits.shape[:-1]
    assert actual.dtype == expected.dtype
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)


@CUDA
@pytest.mark.parametrize("offset, scale", [(0.0, 0.0), (1e6, 1.0)])
def test_cuda_entropy_stable_for_uniform_and_shifted_logits(offset, scale):
    from solution.vocab import entropy

    torch.manual_seed(109)
    logits = torch.randn(3, 4097, device="cuda") * scale + offset
    with torch.no_grad():
        actual = entropy(logits)
        expected = _reference_entropy(logits)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_entropy_grad_enabled_fallback_preserves_derivatives(device):
    from solution.vocab import entropy, supports_entropy

    torch.manual_seed(110)
    logits = torch.randn(2, 3, 4097, device=device, requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    assert torch.is_grad_enabled()
    assert not supports_entropy(logits)
    actual = entropy(logits)
    expected = _reference_entropy(ref_logits)
    assert actual.shape == (2, 3) and actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    upstream = torch.tensor([[-0.7, 0.0, 1.2], [3.25, -2.0, 0.5]], device=device)
    dx = torch.autograd.grad(actual, logits, upstream)[0]
    ref_dx = torch.autograd.grad(expected, ref_logits, upstream)[0]
    torch.testing.assert_close(dx, ref_dx, rtol=5e-4, atol=2e-6)


@CUDA
def test_cuda_entropy_noncontiguous_fallback():
    from solution.vocab import entropy, supports_entropy

    torch.manual_seed(111)
    logits = torch.randn(2, 3, 8194, device="cuda")[..., ::2]
    assert not logits.is_contiguous()
    with torch.no_grad():
        assert not supports_entropy(logits)
        actual = entropy(logits)
        expected = _reference_entropy(logits)
    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)


def test_cpu_fp16_entropy_fallback():
    from solution.vocab import entropy, supports_entropy

    torch.manual_seed(112)
    logits = torch.randn(2, 3, 17, dtype=torch.float16)
    with torch.no_grad():
        assert not supports_entropy(logits)
        actual = entropy(logits)
        expected = _reference_entropy(logits)
    assert actual.dtype == expected.dtype and actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
