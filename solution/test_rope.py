"""Original Qwen2 RoPE versus the fused fp32 forward and transpose derivative."""

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _original(monkeypatch):
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    function = modeling.apply_rotary_pos_emb
    while getattr(function, "_soup_rope_pair_patch", False):
        function = function.__wrapped__
    # Restore the incoming wrapper after the test, even if apply() ran first.
    monkeypatch.setattr(modeling, "apply_rotary_pos_emb", function)
    return modeling, function


def _install(monkeypatch, *, spy_fallback=False):
    from solution.rope import patch_qwen2_rope

    modeling, original = _original(monkeypatch)
    calls = []
    if spy_fallback:
        def observed(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(modeling, "apply_rotary_pos_emb", observed)
    patch_qwen2_rope()
    wrapper = modeling.apply_rotary_pos_emb
    patch_qwen2_rope()
    assert modeling.apply_rotary_pos_emb is wrapper
    return modeling, original, calls


def _inputs(shape, *, device="cuda", dtype=torch.float32, coefficient_batches=(1, 1), strided_coefficients=False):
    batch, qheads, kheads, time, width = shape
    # The real projection storage is B,T,H,D, exposed to RoPE as B,H,T,D.
    q = torch.randn(batch, time, qheads, width, device=device, dtype=dtype).transpose(1, 2).requires_grad_()
    k = torch.randn(batch, time, kheads, width, device=device, dtype=dtype).transpose(1, 2).requires_grad_()

    def coefficient(size):
        # Independent halves catch an incorrect backward that assumes the
        # repeated-frequency pattern produced by the ordinary Qwen model.
        values = torch.randn(size, time * (2 if strided_coefficients else 1), width, device=device, dtype=dtype)
        return values[:, ::2, :] if strided_coefficients else values

    return q, k, coefficient(coefficient_batches[0]), coefficient(coefficient_batches[1])


def _clones(values):
    return [value.detach().clone(memory_format=torch.preserve_format).requires_grad_(value.requires_grad)
            for value in values]


def _upstream(value, *, noncontiguous):
    if noncontiguous:
        gradient = torch.randn(*value.shape[:-1], value.shape[-1] * 2, device=value.device, dtype=value.dtype)[..., ::2]
        assert not gradient.is_contiguous() and gradient.stride(-1) == 2
        return gradient
    return torch.randn_like(value)


@CUDA
@pytest.mark.parametrize("shape, coefficient_batches, strided_coefficients, noncontiguous_grad", [
    ((4, 12, 2, 384, 128), (1, 1), False, False),
    ((2, 3, 1, 17, 64), (2, 2), False, True),
    ((3, 4, 2, 31, 48), (1, 3), True, True),
    ((2, 2, 1, 19, 128), (2, 1), False, False),
    ((1, 1, 1, 1, 2), (1, 1), False, False),
])
def test_cuda_forward_and_all_input_gradients_match_original(
    monkeypatch, shape, coefficient_batches, strided_coefficients, noncontiguous_grad,
):
    from solution.rope import _supported

    modeling, original, _ = _install(monkeypatch)
    torch.manual_seed(241)
    values = _inputs(shape, coefficient_batches=coefficient_batches, strided_coefficients=strided_coefficients)
    reference = _clones(values)
    assert _supported(*values, 1)
    expected = original(*reference)
    actual = modeling.apply_rotary_pos_emb(*values)
    assert all(type(value.grad_fn).__name__ == "_RotaryPairBackward" for value in actual)
    upstream = tuple(_upstream(value, noncontiguous=noncontiguous_grad) for value in actual)
    for output, ref_output in zip(actual, expected):
        assert output.dtype == ref_output.dtype == torch.float32
        torch.testing.assert_close(output, ref_output, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual, values[:2], upstream)
    reference_gradient = torch.autograd.grad(expected, reference[:2], upstream)
    for gradient, ref_gradient in zip(actual_gradient, reference_gradient):
        torch.testing.assert_close(gradient, ref_gradient, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("selected_output", [0, 1])
def test_missing_other_output_gradient_preserves_none_and_selected_derivative(monkeypatch, selected_output):
    modeling, original, _ = _install(monkeypatch)
    torch.manual_seed(242)
    values = _inputs((2, 3, 1, 13, 32), coefficient_batches=(2, 1))
    reference = _clones(values)
    expected, actual = original(*reference), modeling.apply_rotary_pos_emb(*values)
    upstream = _upstream(actual[selected_output], noncontiguous=True)
    actual_gradient = torch.autograd.grad(actual[selected_output], values[:2], upstream, allow_unused=True)
    reference_gradient = torch.autograd.grad(expected[selected_output], reference[:2], upstream, allow_unused=True)
    assert actual_gradient[1 - selected_output] is reference_gradient[1 - selected_output] is None
    torch.testing.assert_close(
        actual_gradient[selected_output], reference_gradient[selected_output], rtol=0, atol=0,
    )


@pytest.mark.parametrize("case, device", [
    ("cpu", "cpu"),
    pytest.param("fp16", "cuda", marks=CUDA),
    pytest.param("custom_layout", "cuda", marks=CUDA),
    pytest.param("unsqueeze_dim", "cuda", marks=CUDA),
    pytest.param("trainable_cos", "cuda", marks=CUDA),
    pytest.param("trainable_sin", "cuda", marks=CUDA),
])
def test_unsupported_calls_use_original_with_its_output_and_gradient_contract(monkeypatch, case, device):
    modeling, original, calls = _install(monkeypatch, spy_fallback=True)
    dtype = torch.float16 if case == "fp16" else torch.float32
    values = list(_inputs((2, 3, 1, 5, 8), device=device, dtype=dtype, coefficient_batches=(2, 2)))
    unsqueeze_dim = 1
    if case == "custom_layout":
        values[:2] = [value.detach().contiguous().requires_grad_() for value in values[:2]]
    elif case == "unsqueeze_dim":
        values[:2] = [value.transpose(1, 2).detach().requires_grad_() for value in values[:2]]
        unsqueeze_dim = 2
    elif case == "trainable_cos":
        values[2].requires_grad_()
    elif case == "trainable_sin":
        values[3].requires_grad_()
    reference = _clones(values)
    expected = original(*reference, unsqueeze_dim=unsqueeze_dim)
    actual = modeling.apply_rotary_pos_emb(*values, unsqueeze_dim=unsqueeze_dim)
    assert len(calls) == 1 and calls[0][1] == {"unsqueeze_dim": unsqueeze_dim}
    upstream = tuple(torch.randn_like(value) for value in expected)
    for output, ref_output in zip(actual, expected):
        torch.testing.assert_close(output, ref_output, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual, [value for value in values if value.requires_grad], upstream)
    reference_gradient = torch.autograd.grad(expected, [value for value in reference if value.requires_grad], upstream)
    for gradient, ref_gradient in zip(actual_gradient, reference_gradient):
        torch.testing.assert_close(gradient, ref_gradient, rtol=0, atol=0)


@CUDA
def test_sequence_length_is_a_runtime_kernel_argument():
    from solution._rope_kernels import rope_pair_forward_kernel, rope_pair_backward_kernel

    for kernel in (rope_pair_forward_kernel, rope_pair_backward_kernel):
        assert "T" in kernel.arg_names
        assert kernel.arg_names.index("T") not in kernel.constexprs, "Each sequence length must not create a new kernel"


@CUDA
def test_actual_nf4_qwen_lora_preserves_loss_all_scaled_gradients_and_rng(monkeypatch):
    from solution import rope
    from solution.test_padded_mlp import (
        _unpatched_model_classes, _make_quantized_lora_model, _batch,
        _rng_state, _restore_rng, _metrics,
    )

    torch.manual_seed(243)
    modeling = _unpatched_model_classes(monkeypatch, v3=True)
    # The baseline must remain the real PyTorch rotary, even when a previous
    # test or solution.apply already installed the fused implementation.
    _original(monkeypatch)
    model, core, trainable = _make_quantized_lora_model(modeling)
    tokens, mask, labels = _batch((11, 7, 2), device="cuda")
    initial_rng = _rng_state()
    with torch.autocast("cuda", dtype=torch.float16):
        expected = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
    expected_gradient = torch.autograd.grad(expected.loss * 128, [parameter for _, parameter in trainable])
    expected_rng = _rng_state()
    expected_metrics = _metrics(expected.logits, mask, labels)
    real_apply = rope._RotaryPair.apply
    calls = []

    def observed_apply(*args):
        calls.append((tuple(args[0].shape), tuple(args[1].shape)))
        return real_apply(*args)

    monkeypatch.setattr(rope._RotaryPair, "apply", observed_apply)
    rope.patch_qwen2_rope()
    _restore_rng(initial_rng)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
    actual_gradient = torch.autograd.grad(actual.loss * 128, [parameter for _, parameter in trainable])
    actual_rng = _rng_state()
    assert len(calls) == len(core.layers), "The full model must actually execute the fused rotary path"
    assert all(query == (3, 4, 11, 32) and key == (3, 2, 11, 32) for query, key in calls)
    assert actual.keys() == expected.keys()
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    for (name, _), gradient, ref_gradient in zip(trainable, actual_gradient, expected_gradient):
        assert torch.isfinite(gradient).all() and ref_gradient.norm() > 0, name
        torch.testing.assert_close(gradient, ref_gradient, rtol=0, atol=0, msg=name)
    for actual_metric, ref_metric in zip(_metrics(actual.logits, mask, labels), expected_metrics):
        torch.testing.assert_close(actual_metric, ref_metric, rtol=0, atol=0)
    assert torch.equal(actual_rng[0], expected_rng[0])
    assert torch.equal(actual_rng[1], expected_rng[1])
