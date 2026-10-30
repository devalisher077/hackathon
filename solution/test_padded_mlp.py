"""End-to-end padding checks with real Qwen2, NF4 and q/v LoRA dropout.

The reference keeps the v3 MLP dtype optimization but processes padded rows.
CUDA tests compare valid outputs, loss/metrics, adapter gradients and the exact
post-forward/backward RNG state. They also observe actual compact projection
inputs, so a silently inactive patch cannot satisfy the fast-path tests.
"""

import sys

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
HIDDEN, INTERMEDIATE, VOCAB = 128, 256, 64


def _without_padding(function):
    markers = ("_soup_padding_mlp_patch", "_soup_padding_linear_patch", "_soup_padded_head_model")
    while any(getattr(function, marker, False) for marker in markers):
        function = function.__wrapped__
    return function


def _unpatched_model_classes(monkeypatch, *, v3=False):
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    for cls in (modeling.Qwen2ForCausalLM, modeling.Qwen2Model, modeling.Qwen2MLP):
        monkeypatch.setattr(cls, "forward", _without_padding(cls.forward))
    # A prior optional integration test may have installed the global bnb
    # wrapper. Strip it for an isolated MLP baseline without importing bnb on
    # CPU-only installations; a previously installed wrapper implies a load.
    bnb_nn = sys.modules.get("bitsandbytes.nn")
    if bnb_nn is not None and hasattr(bnb_nn, "Linear4bit"):
        monkeypatch.setattr(bnb_nn.Linear4bit, "forward", _without_padding(bnb_nn.Linear4bit.forward))
    if v3 and not getattr(modeling.Qwen2MLP.forward, "_soup_mlp_dtype_patch", False):
        pytest.importorskip("triton")
        from solution.swiglu import patch_qwen2_mlp
        from solution.mlp import patch_qwen2_mlp_dtype

        patch_qwen2_mlp()
        patch_qwen2_mlp_dtype()
    return modeling


def _config():
    from transformers import Qwen2Config

    config = Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        hidden_act="silu", attention_dropout=0.0, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    return config


def _make_quantized_lora_model(modeling):
    bnb = pytest.importorskip("bitsandbytes")
    peft = pytest.importorskip("peft")
    base = modeling.Qwen2ForCausalLM(_config())
    core = base.model

    def quantized(original):
        replacement = bnb.nn.Linear4bit(
            original.in_features, original.out_features,
            bias=original.bias is not None, compute_dtype=torch.float16,
            quant_type="nf4", compress_statistics=True,
        )
        with torch.no_grad():
            replacement.weight.copy_(original.weight)
            if original.bias is not None:
                replacement.bias.copy_(original.bias)
        return replacement.half().cuda()

    for layer in core.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, quantized(getattr(layer.self_attn, name)))
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, name, quantized(getattr(layer.mlp, name)))
    base = base.cuda()
    base.is_loaded_in_4bit = True
    peft.prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
        ),
    )
    # Default zero B would make all A gradients zero and weaken this check.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                parameter.normal_(mean=0.0, std=0.02)
    model.train()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 8 and all("lora_" in name for name, _ in trainable)
    assert any(isinstance(layer, torch.nn.Dropout) and layer.p == 0.05 for layer in model.modules())
    assert all(layer.mlp.gate_proj.compute_type_is_set for layer in core.layers)
    return model, core, trainable


def _batch(lengths, *, device):
    width = max(lengths)
    mask = torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    tokens = torch.randint(3, VOCAB, (len(lengths), width), device=device)
    tokens = tokens.masked_fill(~mask, 0)
    labels = tokens.masked_fill(~mask, -100)
    return tokens, mask.to(torch.long), labels


def _rng_state():
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def _restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def _projection_observers(core, destination):
    def observe(module, args):
        destination.append(tuple(args[0].shape))

    return [layer.mlp.gate_proj.register_forward_pre_hook(observe) for layer in core.layers]


def _metrics(logits, mask, labels):
    with torch.no_grad():
        values = logits.float()
        logp = torch.nn.functional.log_softmax(values, dim=-1)
        token_entropy = -(logp.exp() * logp).sum(-1)
        entropy = (token_entropy * mask).sum() / mask.sum()
        targets = labels[:, 1:]
        token_loss = torch.nn.functional.cross_entropy(
            values[:, :-1].reshape(-1, values.shape[-1]), targets.reshape(-1),
            reduction="none", ignore_index=-100,
        ).view_as(targets)
        counts = (targets != -100).sum(-1)
        row_losses = token_loss.sum(-1) / counts.clamp(min=1)
        correct = (values[:, :-1].argmax(-1) == targets) & (targets != -100)
        accuracy = correct.sum() / counts.sum()
    return entropy, accuracy, row_losses, counts


@CUDA
@pytest.mark.parametrize("lengths", [(9, 5, 2), (7, 3)])
def test_compacted_mlp_preserves_valid_outputs_lora_gradients_and_rng(monkeypatch, lengths):
    from solution.padded_mlp import patch_padding_mlp, _PADDING_CONTEXT

    torch.manual_seed(141)
    modeling = _unpatched_model_classes(monkeypatch, v3=True)
    model, core, trainable = _make_quantized_lora_model(modeling)
    tokens, mask, labels = _batch(lengths, device="cuda")
    observed, final_hidden = [], []
    hooks = _projection_observers(core, observed)
    hooks.append(core.norm.register_forward_hook(lambda module, args, output: final_hidden.append(output)))
    initial_rng = _rng_state()
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            expected = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
        reference_hidden = final_hidden[-1].detach().clone()
        reference_logits = expected.logits.detach().clone()
        reference_loss = expected.loss.detach().clone()
        reference_metrics = _metrics(expected.logits, mask, labels)
        reference_gradients = [
            gradient.float() / 128
            for gradient in torch.autograd.grad(expected.loss * 128, [parameter for _, parameter in trainable])
        ]
        reference_rng = _rng_state()
        assert observed == [(len(lengths), max(lengths), HIDDEN)] * len(core.layers)

        observed.clear()
        final_hidden.clear()
        patch_padding_mlp()
        first_model, first_mlp = modeling.Qwen2Model.forward, modeling.Qwen2MLP.forward
        patch_padding_mlp()
        assert modeling.Qwen2Model.forward is first_model
        assert modeling.Qwen2MLP.forward is first_mlp
        _restore_rng(initial_rng)
        with torch.autocast("cuda", dtype=torch.float16):
            actual = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
        actual_gradients = [
            gradient.float() / 128
            for gradient in torch.autograd.grad(actual.loss * 128, [parameter for _, parameter in trainable])
        ]
        actual_rng = _rng_state()
        valid = mask.bool()
        assert observed == [(sum(lengths), HIDDEN)] * len(core.layers)
        assert actual.logits.shape == reference_logits.shape == (*tokens.shape, VOCAB)
        assert final_hidden[-1].shape == reference_hidden.shape == (*tokens.shape, HIDDEN)
        torch.testing.assert_close(final_hidden[-1][valid], reference_hidden[valid], rtol=3e-3, atol=2e-3)
        torch.testing.assert_close(actual.logits[valid], reference_logits[valid], rtol=3e-3, atol=2e-3)
        torch.testing.assert_close(actual.loss, reference_loss, rtol=3e-3, atol=2e-5)
        for (name, _), gradient, expected_gradient in zip(trainable, actual_gradients, reference_gradients):
            assert torch.isfinite(gradient).all() and torch.isfinite(expected_gradient).all(), name
            assert expected_gradient.norm() > 0, name
            # Compaction changes GEMM's M dimension and can select another
            # fp16 reduction order. Compare error to the tensor's scale rather
            # than unstable elementwise relative errors near a zero gradient.
            difference = gradient - expected_gradient
            relative_error = difference.norm() / expected_gradient.norm()
            assert relative_error < 5e-3, (name, float(relative_error))
            max_error = difference.abs().max()
            max_reference = expected_gradient.abs().max()
            assert max_error < 5e-3 * max_reference + 1e-7, (name, float(max_error), float(max_reference))
        actual_metrics = _metrics(actual.logits, mask, labels)
        for actual_metric, expected_metric in zip(actual_metrics[:3], reference_metrics[:3]):
            torch.testing.assert_close(actual_metric, expected_metric, rtol=3e-3, atol=2e-5)
        assert torch.equal(actual_metrics[3], reference_metrics[3])
        assert torch.equal(actual_rng[0], reference_rng[0])
        assert torch.equal(actual_rng[1], reference_rng[1]), "LoRA dropout must consume exactly the original RNG stream"
        assert _PADDING_CONTEXT.get() is None
    finally:
        for hook in hooks:
            hook.remove()


@CUDA
@pytest.mark.parametrize("case", [
    "eval", "left_padding", "interior_padding", "nonbinary", "zero_row", "all_zero",
    "use_cache", "past_key_values", "cache_position", "hidden_states", "no_mask", "dict_mask",
])
def test_unsupported_model_calls_keep_the_full_original_path(monkeypatch, case):
    from solution.padded_mlp import patch_padding_mlp, _PADDING_CONTEXT

    torch.manual_seed(142)
    modeling = _unpatched_model_classes(monkeypatch, v3=True)
    model, core, _ = _make_quantized_lora_model(modeling)
    tokens, mask, _ = _batch((7, 4), device="cuda")
    if case == "eval":
        model.eval()
    elif case == "left_padding":
        mask[1] = torch.tensor([0, 0, 0, 1, 1, 1, 1], device="cuda")
    elif case == "interior_padding":
        mask[1] = torch.tensor([1, 1, 0, 1, 0, 0, 0], device="cuda")
    elif case == "nonbinary":
        mask = mask.float()
        mask[1, 2] = 0.5
    elif case == "zero_row":
        mask[1].zero_()
    elif case == "all_zero":
        mask.zero_()

    def arguments():
        kwargs = {"input_ids": tokens, "attention_mask": mask, "use_cache": False}
        if case == "use_cache":
            kwargs["use_cache"] = True
        elif case == "past_key_values":
            from transformers.cache_utils import DynamicCache

            kwargs["past_key_values"] = DynamicCache(config=core.config)
        elif case == "cache_position":
            kwargs["cache_position"] = torch.arange(tokens.shape[1], device="cuda")
        elif case == "hidden_states":
            kwargs["output_hidden_states"] = True
        elif case == "no_mask":
            kwargs["attention_mask"] = None
        elif case == "dict_mask":
            length = tokens.shape[1]
            causal = torch.ones(length, length, dtype=torch.bool, device="cuda").tril()
            kwargs["attention_mask"] = {"full_attention": causal[None, None, :, :]}
        return kwargs

    observed = []
    hooks = _projection_observers(core, observed)
    initial_rng = _rng_state()
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            expected = model(**arguments()).logits.detach().clone()
        reference_rng = _rng_state()
        observed.clear()
        patch_padding_mlp()
        _restore_rng(initial_rng)
        with torch.autocast("cuda", dtype=torch.float16):
            actual = model(**arguments()).logits
        assert observed == [(2, 7, HIDDEN)] * len(core.layers)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        actual_rng = _rng_state()
        assert torch.equal(actual_rng[0], reference_rng[0])
        assert torch.equal(actual_rng[1], reference_rng[1])
        assert _PADDING_CONTEXT.get() is None
    finally:
        for hook in hooks:
            hook.remove()


def test_cpu_model_calls_are_unchanged_without_gpu_dependencies(monkeypatch):
    from solution.padded_mlp import patch_padding_mlp, _PADDING_CONTEXT

    torch.manual_seed(143)
    modeling = _unpatched_model_classes(monkeypatch)
    model = modeling.Qwen2ForCausalLM(_config()).train()
    tokens, mask, labels = _batch((7, 4), device="cpu")
    expected = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
    patch_padding_mlp()
    actual = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    assert _PADDING_CONTEXT.get() is None


@CUDA
def test_padding_scope_is_reset_after_a_model_exception(monkeypatch):
    from solution.padded_mlp import patch_padding_mlp, _PADDING_CONTEXT

    torch.manual_seed(144)
    modeling = _unpatched_model_classes(monkeypatch, v3=True)
    model, core, _ = _make_quantized_lora_model(modeling)
    tokens, mask, _ = _batch((7, 4), device="cuda")
    patch_padding_mlp()

    def fail_after_mlp(module, args, output):
        assert _PADDING_CONTEXT.get() is not None
        raise RuntimeError("intentional padded-MLP test exception")

    hook = core.layers[0].mlp.register_forward_hook(fail_after_mlp)
    try:
        with pytest.raises(RuntimeError, match="intentional padded-MLP test exception"):
            with torch.autocast("cuda", dtype=torch.float16):
                model(input_ids=tokens, attention_mask=mask, use_cache=False)
    finally:
        hook.remove()
    assert _PADDING_CONTEXT.get() is None
    observed = []
    hooks = _projection_observers(core, observed)
    try:
        x = torch.randn(2, 7, HIDDEN, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = core.layers[0].mlp(x)
        assert output.shape == x.shape and observed == [(2, 7, HIDDEN)]
    finally:
        for hook in hooks:
            hook.remove()
