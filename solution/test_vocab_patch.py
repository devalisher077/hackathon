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


def _install_causal_patch(monkeypatch, module):
    from solution.vocab_patch import _patch_causal_lm_loss

    # ForCausalLMLoss resolves fixed_cross_entropy through module globals.
    # Keep both reference functions plain even when apply() ran beforehand.
    monkeypatch.setattr(module, "fixed_cross_entropy", _unpatched(module, "fixed_cross_entropy"))
    original = _unpatched(module, "ForCausalLMLoss")
    monkeypatch.setattr(module, "ForCausalLMLoss", original)
    aliases = []
    for name, function in list(module.LOSS_MAPPING.items()):
        while getattr(function, "_soup_vocab_patch", False):
            function = function.__wrapped__
        if function is original:
            aliases.append(name)
            monkeypatch.setitem(module.LOSS_MAPPING, name, original)
    assert "ForCausalLM" in aliases
    _patch_causal_lm_loss(module)
    first = module.ForCausalLMLoss
    _patch_causal_lm_loss(module)
    assert module.ForCausalLMLoss is first
    assert all(module.LOSS_MAPPING[name] is first for name in aliases)
    return original


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_causal_loss_cpu_preserves_upcast_alignment_and_mapping_aliases(monkeypatch, dtype):
    module = pytest.importorskip("transformers.loss.loss_utils")
    original = _install_causal_patch(monkeypatch, module)
    torch.manual_seed(78)
    logits = torch.randn(2, 4, 17, dtype=dtype, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[-100, 1, 2, -100], [0, 3, 16, 7]])
    expected = original(reference, labels, 17, num_items_in_batch=torch.tensor(9))
    actual = module.ForCausalLMLoss(logits, labels, 17, num_items_in_batch=torch.tensor(9))
    assert actual.dtype == expected.dtype == torch.float32
    _compare_scalar_loss_and_grad(actual, expected, logits, reference)


@CUDA
@pytest.mark.parametrize(
    "dtype, denominator, supplied_shift",
    [(torch.float16, None, False), (torch.float16, 13, False),
     (torch.float16, torch.tensor(7), True), (torch.float32, 11, True)],
)
def test_cuda_causal_loss_uses_original_logit_storage_and_scaled_gradient(
    monkeypatch, dtype, denominator, supplied_shift,
):
    module = pytest.importorskip("transformers.loss.loss_utils")
    from solution import vocab

    original = _install_causal_patch(monkeypatch, module)
    real_ce = vocab.per_token_cross_entropy_fp32
    sources = []

    def observed_ce(source, *args, **kwargs):
        sources.append((source.dtype, source.untyped_storage().data_ptr()))
        return real_ce(source, *args, **kwargs)

    monkeypatch.setattr(vocab, "per_token_cross_entropy_fp32", observed_ce)
    torch.manual_seed(79)
    logits = torch.randn(2, 5, 151936, device="cuda", dtype=dtype, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[-100, 1, 151935, -100, 7], [0, 3, 4, 5, 6]], device="cuda")
    kwargs = {"num_items_in_batch": denominator}
    if supplied_shift:
        # Explicitly supplied alignment must win over labels; shifting twice
        # would change both the loss and every corresponding gradient row.
        kwargs["shift_labels"] = torch.tensor(
            [[6, -100, 3, 2, 1], [4, 8, -100, 7, -100]], device="cuda"
        )
    expected = original(reference, labels, 151936, **kwargs)
    actual = module.ForCausalLMLoss(logits, labels, 151936, **kwargs)
    assert sources == [(dtype, logits.untyped_storage().data_ptr())]
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    # GradScaler multiplies before the fp32 -> fp16 gradient cast. Applying
    # its multiplier after that cast would silently lose this equivalence.
    dx = torch.autograd.grad(actual * 32768.0, logits)[0]
    ref_dx = torch.autograd.grad(expected * 32768.0, reference)[0]
    assert dx.dtype == dtype
    torch.testing.assert_close(dx, ref_dx, rtol=2e-3 if dtype == torch.float16 else 3e-4, atol=2e-6)


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


@CUDA
def test_cuda_causal_loss_inside_real_fp16_autocast_matches_plain_reference(monkeypatch):
    module = pytest.importorskip("transformers.loss.loss_utils")
    from solution import vocab

    # This helper unwraps both ForCausalLMLoss and fixed_cross_entropy, so
    # the expected result cannot accidentally pass through our kernels.
    original = _install_causal_patch(monkeypatch, module)
    real_ce = vocab.per_token_cross_entropy_fp32
    fast_input_dtypes = []

    def observed_ce(source, *args, **kwargs):
        fast_input_dtypes.append(source.dtype)
        return real_ce(source, *args, **kwargs)

    monkeypatch.setattr(vocab, "per_token_cross_entropy_fp32", observed_ce)
    torch.manual_seed(80)
    hidden = torch.randn(2, 5, 32, device="cuda", requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    frozen_head = torch.randn(4097, 32, device="cuda") / (32 ** 0.5)
    labels = torch.tensor([[-100, 1, 4096, -100, 7], [0, 3, 4, 5, 6]], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = hidden @ frozen_head.t()
        reference_logits = reference_hidden @ frozen_head.t()
        assert logits.dtype == reference_logits.dtype == torch.float16
        expected = original(reference_logits, labels, 4097, num_items_in_batch=11)
        actual = module.ForCausalLMLoss(logits, labels, 4097, num_items_in_batch=11)
    assert fast_input_dtypes == [torch.float16]
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    gradients = torch.autograd.grad(actual * 32768.0, (logits, hidden))
    reference_gradients = torch.autograd.grad(
        expected * 32768.0, (reference_logits, reference_hidden)
    )
    assert gradients[0].dtype == torch.float16 and gradients[1].dtype == torch.float32
    torch.testing.assert_close(gradients[0], reference_gradients[0], rtol=2e-3, atol=2e-6)
    # The head's fp16 matmul adds its own accumulation and rounding; verify
    # that the scaled CE gradient also propagates through this real AMP path.
    torch.testing.assert_close(gradients[1], reference_gradients[1], rtol=3e-3, atol=5e-2)
