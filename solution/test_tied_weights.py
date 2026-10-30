"""Storage, output, AMP-gradient, RNG and setup contracts for tied weights."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


class _TiedModel(torch.nn.Module):
    def __init__(self, device="cpu", vocab=257, width=64):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen2")
        self.embedding = torch.nn.Embedding(vocab, width, device=device)
        self.head = torch.nn.Linear(width, vocab, bias=False, device=device)
        self.embedding.weight.data = self.embedding.weight.data.half().float()
        self.embedding.weight.requires_grad_(False)
        self.head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


def test_lossless_copy_preserves_values_and_rng_on_cpu():
    from solution.tied_weights import _lossless_half_copy

    weight = torch.randn(257, 64).half().float()
    before = weight.clone()
    state = torch.get_rng_state().clone()
    half = _lossless_half_copy(weight)
    assert half is not None and half.dtype == torch.float16
    torch.testing.assert_close(half.float(), weight, rtol=0, atol=0)
    torch.testing.assert_close(weight, before, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("value", [0.123456789, float("inf"), float("nan")])
def test_lossless_copy_rejects_changed_or_nonfinite_values(value):
    from solution.tied_weights import _lossless_half_copy

    weight = torch.ones(7, 11)
    weight[2, 3] = value
    assert _lossless_half_copy(weight) is None


def test_cpu_model_is_unchanged():
    from solution.tied_weights import compress_tied_weights

    model = _TiedModel()
    parameter = model.embedding.weight
    pointer = parameter.data_ptr()
    assert not compress_tied_weights(model)
    assert model.embedding.weight is model.head.weight is parameter
    assert parameter.dtype == torch.float32 and parameter.data_ptr() == pointer


def test_setup_wrapper_preserves_return_arguments_and_fake_models(monkeypatch):
    import solution.tied_weights as implementation

    calls = []
    sentinel = object()

    class FakeWrapper:
        def _setup_transformers(self, *args, **kwargs):
            calls.append((args, kwargs))
            self.model = SimpleNamespace()
            return sentinel

    fake_module = SimpleNamespace(SFTTrainerWrapper=FakeWrapper)

    def load_module(name):
        assert name == "soup_cli.trainer.sft"
        return fake_module

    monkeypatch.setattr(implementation, "import_module", load_module)
    implementation.patch_tied_weights()
    first_patch = FakeWrapper._setup_transformers
    implementation.patch_tied_weights()
    assert FakeWrapper._setup_transformers is first_patch
    wrapper = FakeWrapper()
    assert wrapper._setup_transformers("cfg", training="tcfg") is sentinel
    assert calls == [(("cfg",), {"training": "tcfg"})]


@CUDA
@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_storage_output_and_scaled_input_gradient_are_exact(autocast):
    from solution.tied_weights import compress_tied_weights

    torch.manual_seed(101)
    model = _TiedModel("cuda")
    parameter = model.embedding.weight
    metadata = [(name, id(p), tuple(p.shape), p.requires_grad) for name, p in model.named_parameters()]
    old_bytes = parameter.untyped_storage().nbytes()
    ids = torch.tensor([[0, 256, 17], [4, 4, 99]], device="cuda")
    expected_embedding = model.embedding(ids)
    reference_x = torch.randn(2, 3, 64, device="cuda", requires_grad=True)
    upstream = torch.randn(2, 3, 257, device="cuda") * 32

    def amp():
        return torch.autocast("cuda", dtype=torch.float16) if autocast else nullcontext()

    with amp():
        expected = model.head(reference_x)
    expected_dx = torch.autograd.grad(expected, reference_x, upstream.to(expected.dtype))[0]
    allocated_before = torch.cuda.memory_allocated()
    assert compress_tied_weights(model)
    allocated_after = torch.cuda.memory_allocated()
    assert allocated_after < allocated_before
    assert model.embedding.weight is model.head.weight is parameter
    assert parameter.untyped_storage().nbytes() * 2 == old_bytes
    assert metadata == [(name, id(p), tuple(p.shape), p.requires_grad) for name, p in model.named_parameters()]
    pointer = parameter.data_ptr()
    assert compress_tied_weights(model) and parameter.data_ptr() == pointer
    actual_embedding = model.embedding(ids)
    assert actual_embedding.dtype == torch.float32
    torch.testing.assert_close(actual_embedding, expected_embedding, rtol=0, atol=0)
    x = reference_x.detach().clone().requires_grad_(True)
    with amp():
        actual = model.head(x)
    dx = torch.autograd.grad(actual, x, upstream.to(actual.dtype))[0]
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
    assert parameter.grad is None and not parameter.requires_grad


@CUDA
def test_nonrepresentable_weight_leaves_cuda_model_untouched():
    from solution.tied_weights import compress_tied_weights

    model = _TiedModel("cuda")
    with torch.no_grad():
        model.embedding.weight[0, 0] = 0.123456789
    parameter = model.embedding.weight
    pointer = parameter.data_ptr()
    assert not compress_tied_weights(model)
    assert parameter.dtype == torch.float32 and parameter.data_ptr() == pointer
    assert getattr(model.embedding.forward, "__func__", None) is torch.nn.Embedding.forward
    assert getattr(model.head.forward, "__func__", None) is torch.nn.Linear.forward


@CUDA
def test_existing_hooks_dropout_rng_and_trainable_gradients_are_preserved():
    from solution.tied_weights import compress_tied_weights

    torch.manual_seed(102)
    model = _TiedModel("cuda")
    projection = torch.nn.Linear(64, 64, bias=False, device="cuda")
    dropout = torch.nn.Dropout(0.05)
    ids = torch.tensor([[0, 13, 17], [2, 99, 256]], device="cuda")
    events = []

    def before_embedding(module, args):
        events.append(("embedding-pre", args[0].dtype))

    def after_embedding(module, args, output):
        events.append(("embedding-post", output.dtype))
        output.requires_grad_(True)

    def after_head(module, args, output):
        events.append(("head-post", output.dtype))

    model.embedding.register_forward_pre_hook(before_embedding)
    model.embedding.register_forward_hook(after_embedding)
    model.head.register_forward_hook(after_head)

    def run():
        with torch.autocast("cuda", dtype=torch.float16):
            embedded = model.embedding(ids)
            assert embedded.dtype == torch.float32 and embedded.requires_grad
            hidden = dropout(projection(embedded).float())
            output = model.head(hidden)
            loss = output.float().square().mean() * 128
        gradient = torch.autograd.grad(loss, projection.weight)[0]
        return output, gradient

    start_rng = torch.cuda.get_rng_state().clone()
    expected, expected_grad = run()
    expected_events = list(events)
    end_rng = torch.cuda.get_rng_state().clone()
    before_compression_rng = torch.cuda.get_rng_state().clone()
    assert compress_tied_weights(model)
    assert torch.equal(before_compression_rng, torch.cuda.get_rng_state())
    torch.cuda.set_rng_state(start_rng)
    events.clear()
    actual, gradient = run()
    assert events == expected_events
    assert torch.equal(end_rng, torch.cuda.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_grad, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("unsupported", ["trainable", "untied", "bias", "max_norm", "custom_forward"])
def test_unsupported_pair_is_not_modified(unsupported):
    from solution.tied_weights import compress_tied_weights

    model = _TiedModel("cuda")
    if unsupported == "trainable":
        model.embedding.weight.requires_grad_(True)
    elif unsupported == "untied":
        model.head.weight = torch.nn.Parameter(model.embedding.weight.detach().clone(), requires_grad=False)
    elif unsupported == "bias":
        model.head.bias = torch.nn.Parameter(torch.zeros(257, device="cuda"), requires_grad=False)
    elif unsupported == "max_norm":
        model.embedding.max_norm = 1.0
    elif unsupported == "custom_forward":
        original = model.head.forward
        model.head.forward = lambda input: original(input) + 1
    parameter = model.embedding.weight
    pointer = parameter.data_ptr()
    assert not compress_tied_weights(model)
    assert model.embedding.weight is parameter
    assert parameter.dtype == torch.float32 and parameter.data_ptr() == pointer


@CUDA
def test_tiny_qwen_preserves_all_logits_loss_and_embedding_input_gradient():
    transformers = pytest.importorskip("transformers")
    from solution.tied_weights import compress_tied_weights

    torch.manual_seed(103)
    config = transformers.Qwen2Config(
        vocab_size=257, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True, attention_dropout=0.0, use_cache=False,
    )
    model = transformers.Qwen2ForCausalLM(config).cuda().half().float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_input_require_grads()
    model.train()
    ids = torch.tensor([[0, 13, 5, 19], [2, 17, 101, 256]], device="cuda")
    labels = ids.clone()
    labels[:, 0] = -100
    captured = []
    model.get_input_embeddings().register_forward_hook(lambda module, args, output: captured.append(output))

    with torch.autocast("cuda", dtype=torch.float16):
        expected = model(input_ids=ids, labels=labels, output_hidden_states=True)
    expected_dx = torch.autograd.grad(expected.loss * 128, captured[-1])[0]
    assert compress_tied_weights(model)
    captured.clear()
    with torch.autocast("cuda", dtype=torch.float16):
        actual = model(input_ids=ids, labels=labels, output_hidden_states=True)
    dx = torch.autograd.grad(actual.loss * 128, captured[-1])[0]
    assert actual.keys() == expected.keys()
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
    for actual_hidden, expected_hidden in zip(actual.hidden_states, expected.hidden_states):
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
