"""Optional attention-weight cache: invalidation, exact bnb math, LoRA and RNG."""

from types import SimpleNamespace

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_cpu_unknown_models_and_metadata_invalidation():
    from solution.attention_cache import cache_attention_weights, _cached_weight

    assert cache_attention_weights(None) == 0
    model = torch.nn.Linear(8, 4)
    model.config = SimpleNamespace(model_type="qwen2")
    assert cache_attention_weights(model) == 0 and "forward" not in model.__dict__
    weight = torch.zeros(16, 1, dtype=torch.uint8)
    nested = SimpleNamespace(absmax=torch.ones(1), code=torch.ones(256),
                             blocksize=256, dtype=torch.float32, nested=False)
    weight.quant_state = SimpleNamespace(absmax=torch.zeros(1, dtype=torch.uint8),
                                         offset=torch.zeros(()), state2=nested,
                                         shape=(4, 8), dtype=torch.float16,
                                         blocksize=64, quant_type="nf4", nested=True)
    layer = SimpleNamespace(weight=weight)
    decodes = []

    def decode(packed, state):
        decodes.append(1)
        return torch.full(state.shape, state.offset.item(), dtype=torch.float16)

    first = _cached_weight(layer, decode)
    assert _cached_weight(layer, decode) is first and len(decodes) == 1
    weight.quant_state.offset.add_(1)
    assert _cached_weight(layer, decode) is not first and len(decodes) == 2
    weight.add_(1)
    _cached_weight(layer, decode)
    assert len(decodes) == 3
    weight.quant_state.state2 = SimpleNamespace(**vars(nested))
    _cached_weight(layer, decode)
    assert len(decodes) == 4


def test_setup_wrapper_preserves_chain_arguments_and_idempotence(monkeypatch):
    import solution.attention_cache as implementation

    calls, sentinel = [], object()

    class Wrapper:
        def _setup_transformers(self, *args, **kwargs):
            calls.append((args, kwargs))
            self.model = object()
            return sentinel

    monkeypatch.setattr(implementation, "import_module",
                        lambda name: SimpleNamespace(SFTTrainerWrapper=Wrapper))
    monkeypatch.setattr(implementation, "cache_attention_weights",
                        lambda model, projections: calls.append((model, projections)))
    implementation.patch_attention_cache(("k_proj", "v_proj"))
    original = Wrapper._setup_transformers
    implementation.patch_attention_cache()
    assert Wrapper._setup_transformers is original
    wrapper = Wrapper()
    assert wrapper._setup_transformers(12, option=True) is sentinel
    assert calls == [((12,), {"option": True}), (wrapper.model, ("k_proj", "v_proj"))]


def _linear():
    bnb = pytest.importorskip("bitsandbytes")
    layer = bnb.nn.Linear4bit(128, 64, bias=True, compute_dtype=torch.float16,
                              quant_type="nf4", compress_statistics=True)
    layer.weight = bnb.nn.Params4bit(torch.randn(64, 128).half() * 0.04,
                                     requires_grad=False, quant_type="nf4", compress_statistics=True)
    layer = layer.cuda().train()
    layer.requires_grad_(False)
    return bnb, layer


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_dense_cache_matches_real_bnb_repeated_forward_and_scaled_backward(monkeypatch, dtype):
    from solution.attention_cache import _install_projection, _CACHE_ATTR

    torch.manual_seed(201)
    bnb, layer = _linear()
    x = torch.randn(2, 1024, 128, device="cuda", dtype=dtype, requires_grad=True)
    upstream = torch.randn(2, 1024, 64, device="cuda", dtype=dtype) * 16
    with torch.autocast("cuda", dtype=torch.float16):
        reference = layer(x)
    reference_dx = torch.autograd.grad(reference, x, upstream)[0]
    reference_bias_dtype = layer.bias.dtype
    reference_bias_value = layer.bias.detach().clone()
    pointer, state = layer.weight.data_ptr(), layer.weight.quant_state
    original = bnb.functional.dequantize_4bit
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(bnb.functional, "dequantize_4bit", observed)
    assert _install_projection(layer, bnb.nn.Linear4bit)
    assert not _install_projection(layer, bnb.nn.Linear4bit)
    rng = torch.cuda.get_rng_state().clone()
    for _ in range(2):
        actual_x = x.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = layer(actual_x)
        gradient = torch.autograd.grad(output, actual_x, upstream)[0]
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        torch.testing.assert_close(gradient, reference_dx, rtol=0, atol=0)
    assert calls == [1]
    assert layer.weight.data_ptr() == pointer and layer.weight.quant_state is state
    assert layer.bias.dtype == reference_bias_dtype
    torch.testing.assert_close(layer.bias, reference_bias_value, rtol=0, atol=0)
    assert getattr(layer, _CACHE_ATTR).weight.numel() * 2 == 64 * 128 * 2
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert all("_soup_attention" not in key for key in layer.state_dict())


@CUDA
def test_dense_cache_invalidates_quantization_and_falls_back_for_native_custom_gemm():
    from solution.attention_cache import _install_projection, _CACHE_ATTR

    torch.manual_seed(202)
    bnb, layer = _linear()
    plain = layer.forward
    assert _install_projection(layer, bnb.nn.Linear4bit)
    x = torch.randn(2048, 128, device="cuda", requires_grad=True)
    layer(x)
    first = getattr(layer, _CACHE_ATTR).weight
    layer.weight.quant_state.offset.add_(0.015625)
    expected = plain(x)
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert getattr(layer, _CACHE_ATTR).weight is not first
    tiny = x[:1]
    expected_tiny = plain(tiny)
    actual_tiny = layer(tiny)
    torch.testing.assert_close(actual_tiny, expected_tiny, rtol=0, atol=0)
    assert getattr(layer, _CACHE_ATTR) is None
    layer.eval()
    with torch.no_grad():
        torch.testing.assert_close(layer(x), plain(x), rtol=0, atol=0)
    assert getattr(layer, _CACHE_ATTR) is None


@CUDA
def test_dense_cache_rebuilds_on_a_different_cuda_stream():
    from solution.attention_cache import _install_projection, _CACHE_ATTR

    bnb, layer = _linear()
    assert _install_projection(layer, bnb.nn.Linear4bit)
    x = torch.randn(2048, 128, device="cuda", requires_grad=True)
    first_output = layer(x)
    first_weight = getattr(layer, _CACHE_ATTR).weight
    other = torch.cuda.Stream()
    other.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other):
        other_output = layer(x)
        other_weight = getattr(layer, _CACHE_ATTR).weight
    torch.cuda.current_stream().wait_stream(other)
    assert other_weight is not first_weight
    torch.testing.assert_close(other_output, first_output, rtol=0, atol=0)


@CUDA
def test_full_qwen_lora_dropout_metrics_gradients_and_state_dict_are_preserved(monkeypatch):
    from solution.test_padded_mlp import (
        _make_quantized_lora_model, _unpatched_model_classes, _batch, _metrics,
        _rng_state, _restore_rng,
    )
    from solution.attention_cache import cache_attention_weights, _CACHE_ATTR

    torch.manual_seed(203)
    modeling = _unpatched_model_classes(monkeypatch, v3=True)
    model, core, trainable = _make_quantized_lora_model(modeling)
    # More than bnb's custom-GEMM row ceiling, even on GPU architectures whose
    # tiny-matrix heuristics differ from T4. The token order/masks stay intact.
    tokens, mask, labels = _batch((513, 495, 463), device="cuda")
    rng = _rng_state()
    with torch.autocast("cuda", dtype=torch.float16):
        reference = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
    expected_logits = reference.logits.detach().clone()
    expected_loss = reference.loss.detach().clone()
    expected_metrics = _metrics(reference.logits, mask, labels)
    expected_gradients = torch.autograd.grad(reference.loss * 16, [p for _, p in trainable])
    expected_rng = _rng_state()
    keys = list(model.state_dict())
    assert cache_attention_weights(model) == 4 * len(core.layers)
    assert cache_attention_weights(model) == 0
    _restore_rng(rng)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
    gradients = torch.autograd.grad(actual.loss * 16, [p for _, p in trainable])
    actual_rng = _rng_state()
    assert torch.equal(actual_rng[0], expected_rng[0]) and torch.equal(actual_rng[1], expected_rng[1])
    torch.testing.assert_close(actual.logits, expected_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual.loss, expected_loss, rtol=1e-5, atol=1e-6)
    for (name, _), gradient, expected in zip(trainable, gradients, expected_gradients):
        assert torch.isfinite(gradient).all() and expected.norm() > 0, name
        torch.testing.assert_close(gradient, expected, rtol=1e-4, atol=1e-6)
    for actual_metric, expected in zip(_metrics(actual.logits, mask, labels), expected_metrics):
        torch.testing.assert_close(actual_metric, expected, rtol=1e-5, atol=1e-6)
    assert list(model.state_dict()) == keys
    for layer in core.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            projection = getattr(layer.self_attn, name)
            base = getattr(projection, "base_layer", projection)
            assert getattr(base, _CACHE_ATTR).weight.dtype == torch.float16
