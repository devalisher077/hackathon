"""Compare patched integration points with their real, unpatched references.

CUDA cases execute the numerical implementation and must pass on Kaggle T4.
CPU fallback tests alone do not validate the Triton kernels.
"""

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _unpatched(module, name):
    function = getattr(module, name)
    while getattr(function, "_soup_vocab_patch", False):
        function = function.__wrapped__
    return function


def _install_with_fallback_spy(monkeypatch, module, name, patch):
    original = _unpatched(module, name)
    calls = []

    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, observed)
    patch(module)
    first = getattr(module, name)
    patch(module)
    assert getattr(module, name) is first, "Repeated patch must not nest wrappers"
    return original, calls


def _compare_scalar_loss_and_grad(actual, expected, x, ref_x):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
    actual_grad = torch.autograd.grad(actual, x)[0]
    expected_grad = torch.autograd.grad(expected, ref_x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-4, atol=2e-6)


@pytest.mark.parametrize("denominator", [None, 11, torch.tensor(7)])
def test_fixed_cross_entropy_cpu_uses_original(monkeypatch, denominator):
    module = pytest.importorskip("transformers.loss.loss_utils")
    from solution.vocab_patch import _patch_fixed_cross_entropy

    original, calls = _install_with_fallback_spy(
        monkeypatch, module, "fixed_cross_entropy", _patch_fixed_cross_entropy
    )
    torch.manual_seed(71)
    x = torch.randn(4, 17, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    labels = torch.tensor([-100, 0, 16, 4])
    expected = original(ref_x, labels, num_items_in_batch=denominator)
    actual = module.fixed_cross_entropy(x, labels, num_items_in_batch=denominator)
    assert len(calls) == 1
    _compare_scalar_loss_and_grad(actual, expected, x, ref_x)


def test_entropy_cpu_preserves_reference_and_gradients(monkeypatch):
    module = pytest.importorskip("trl.trainer.sft_trainer")
    from solution.vocab_patch import _patch_entropy

    original, calls = _install_with_fallback_spy(
        monkeypatch, module, "entropy_from_logits", _patch_entropy
    )
    torch.manual_seed(72)
    x = torch.randn(2, 5, 17, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    expected = original(ref_x, chunk_size=3)
    actual = module.entropy_from_logits(x, chunk_size=3)
    assert len(calls) == 1 and calls[0][1]["chunk_size"] == 3
    assert actual.shape == x.shape[:-1]
    _compare_scalar_loss_and_grad(actual.sum(), expected.sum(), x, ref_x)


def _rewind_labels(device):
    return torch.tensor(
        [[-100, -100, -100, -100, -100, -100],
         [-100, 0, 3, -100, 7, 16],
         [11, 1, 2, 3, 4, 5]],
        device=device,
        dtype=torch.long,
    )


def test_rewind_cpu_uses_original_and_preserves_empty_rows(monkeypatch):
    module = pytest.importorskip("soup_cli.trainer.rewind_hf")
    from solution.vocab_patch import _patch_rewind

    original, calls = _install_with_fallback_spy(
        monkeypatch, module, "row_losses", _patch_rewind
    )
    torch.manual_seed(73)
    logits = torch.randn(3, 6, 17, requires_grad=True)
    labels = _rewind_labels("cpu")
    expected_losses, expected_counts = original(logits, labels)
    losses, counts = module.row_losses(logits, labels)
    assert len(calls) == 1
    assert counts == expected_counts == [0, 4, 5]
    assert losses == pytest.approx(expected_losses)
    assert losses[0] == 0.0


@CUDA
@pytest.mark.parametrize("denominator", [None, 13, torch.tensor(7)])
def test_cuda_fixed_loss_calls_real_kernel_and_preserves_denominator(monkeypatch, denominator):
    module = pytest.importorskip("transformers.loss.loss_utils")
    from solution import vocab
    from solution.vocab_patch import _patch_fixed_cross_entropy

    original, fallback_calls = _install_with_fallback_spy(
        monkeypatch, module, "fixed_cross_entropy", _patch_fixed_cross_entropy
    )
    real_ce = vocab.per_token_cross_entropy
    fast_calls = []

    def observed_ce(*args, **kwargs):
        result = real_ce(*args, **kwargs)
        fast_calls.append(result)
        return result

    monkeypatch.setattr(vocab, "per_token_cross_entropy", observed_ce)
    torch.manual_seed(74)
    x = torch.randn(5, 151936, device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    labels = torch.tensor([-100, 0, 151935, 1234, -100], device="cuda")
    assert vocab.supports_cross_entropy(x, labels)
    expected = original(ref_x, labels, num_items_in_batch=denominator)
    actual = module.fixed_cross_entropy(x, labels, num_items_in_batch=denominator)
    assert not fallback_calls and len(fast_calls) == 1
    assert fast_calls[0].requires_grad
    _compare_scalar_loss_and_grad(actual, expected, x, ref_x)


@CUDA
def test_cuda_no_grad_entropy_calls_real_kernel(monkeypatch):
    module = pytest.importorskip("trl.trainer.sft_trainer")
    from solution import vocab
    from solution.vocab_patch import _patch_entropy

    original, fallback_calls = _install_with_fallback_spy(
        monkeypatch, module, "entropy_from_logits", _patch_entropy
    )
    real_entropy = vocab.entropy
    fast_calls = []

    def observed_entropy(*args, **kwargs):
        result = real_entropy(*args, **kwargs)
        fast_calls.append(result)
        return result

    monkeypatch.setattr(vocab, "entropy", observed_entropy)
    torch.manual_seed(75)
    logits = (torch.randn(2, 5, 151936, device="cuda") * 3).requires_grad_(True)
    with torch.no_grad():
        assert vocab.supports_entropy(logits)
        expected = original(logits, chunk_size=3)
        actual = module.entropy_from_logits(logits, chunk_size=3)
    assert not fallback_calls and len(fast_calls) == 1
    assert actual.shape == logits.shape[:-1]
    assert actual.dtype == expected.dtype and not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-5)


@CUDA
def test_cuda_entropy_with_gradients_uses_original(monkeypatch):
    module = pytest.importorskip("trl.trainer.sft_trainer")
    from solution.vocab_patch import _patch_entropy

    original, calls = _install_with_fallback_spy(
        monkeypatch, module, "entropy_from_logits", _patch_entropy
    )
    torch.manual_seed(76)
    x = torch.randn(2, 5, 17, device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    expected = original(ref_x, chunk_size=3)
    actual = module.entropy_from_logits(x, chunk_size=3)
    assert len(calls) == 1
    _compare_scalar_loss_and_grad(actual.sum(), expected.sum(), x, ref_x)


@CUDA
def test_cuda_rewind_calls_real_kernel_and_keeps_counts(monkeypatch):
    module = pytest.importorskip("soup_cli.trainer.rewind_hf")
    from solution import vocab
    from solution.vocab_patch import _patch_rewind

    original, fallback_calls = _install_with_fallback_spy(
        monkeypatch, module, "row_losses", _patch_rewind
    )
    real_ce = vocab.per_token_cross_entropy
    fast_calls = []

    def observed_ce(*args, **kwargs):
        result = real_ce(*args, **kwargs)
        fast_calls.append(result)
        return result

    monkeypatch.setattr(vocab, "per_token_cross_entropy", observed_ce)
    torch.manual_seed(77)
    logits = torch.randn(3, 6, 151936, device="cuda", requires_grad=True)
    labels = _rewind_labels("cuda")
    expected_losses, expected_counts = original(logits, labels)
    losses, counts = module.row_losses(logits, labels)
    assert not fallback_calls and len(fast_calls) == 3
    assert all(not value.requires_grad for value in fast_calls)
    assert counts == expected_counts == [0, 4, 5]
    assert losses[0] == 0.0
    assert losses == pytest.approx(expected_losses, rel=1e-4, abs=5e-5)
