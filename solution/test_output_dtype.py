"""Lossless training-logit storage through real Accelerate, TRL and Soup."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _conversion_patch(monkeypatch):
    operations = pytest.importorskip("accelerate.utils.operations")
    from solution.output_dtype import patch_output_dtype

    original = operations.ConvertOutputsToFp32.__call__
    while getattr(original, "_soup_scoped_half_logits", False):
        original = original.__wrapped__
    monkeypatch.setattr(operations.ConvertOutputsToFp32, "__call__", original)
    assert patch_output_dtype()
    wrapper = operations.ConvertOutputsToFp32.__call__
    assert patch_output_dtype() and operations.ConvertOutputsToFp32.__call__ is wrapper
    return operations, original, wrapper


def _training_patches(monkeypatch):
    trl = pytest.importorskip("trl.trainer.sft_trainer")
    losses = pytest.importorskip("transformers.loss.loss_utils")
    rewind = pytest.importorskip("soup_cli.trainer.rewind_hf")
    from solution.metrics import patch_sft_metrics
    from solution.vocab_patch import _patch_causal_lm_loss, _patch_entropy, _patch_rewind

    def reset(owner, name, marker):
        function = getattr(owner, name)
        while getattr(function, marker, False):
            function = function.__wrapped__
        monkeypatch.setattr(owner, name, function)

    reset(trl.SFTTrainer, "compute_loss", "_soup_no_logits_metric_copy")
    reset(trl, "entropy_from_logits", "_soup_vocab_patch")
    reset(rewind, "row_losses", "_soup_vocab_patch")
    reset(losses, "ForCausalLMLoss", "_soup_vocab_patch")
    for name, function in list(losses.LOSS_MAPPING.items()):
        while getattr(function, "_soup_vocab_patch", False):
            function = function.__wrapped__
        monkeypatch.setitem(losses.LOSS_MAPPING, name, function)
    _patch_causal_lm_loss(losses)
    _patch_entropy(trl)
    _patch_rewind(rewind)
    assert patch_sft_metrics()
    return trl, rewind


def _trainer(cls, model, accelerator):
    trainer = object.__new__(cls)
    trainer.model = model
    trainer.args = SimpleNamespace(
        use_liger_kernel=False, loss_type="nll", average_tokens_across_devices=False,
        n_gpu=1, fp16=True, bf16=False,
    )
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = False
    trainer.padding_free = False
    trainer.compute_loss_func = None
    trainer.label_smoother = None
    trainer.model_accepts_loss_kwargs = True
    trainer.accelerator = accelerator
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._total_train_tokens = 37
    trainer.state = SimpleNamespace(global_step=4)
    return trainer


def _model(*, vocab=257):
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=vocab, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0.0, tie_word_embeddings=True, use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = transformers.Qwen2ForCausalLM(config).cuda().half().float()
    return model.train()


def _batch(*, vocab=257, batch=2, length=7):
    ids = torch.arange(batch * length, device="cuda").reshape(batch, length) % vocab
    mask = torch.ones_like(ids)
    mask[-1, -2:] = 0
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :2] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _rng():
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def _restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_and_unscoped_outputs_keep_real_accelerate_conversion(monkeypatch, dtype):
    operations, original, _ = _conversion_patch(monkeypatch)
    from solution.output_dtype import _CONTEXT, _Scope, is_preserved_logits
    from transformers.modeling_outputs import CausalLMOutputWithPast

    class Model(torch.nn.Module):
        def forward(self):
            return CausalLMOutputWithPast(
                loss=torch.ones((), requires_grad=True),
                logits=torch.ones(2, 3, 5, dtype=dtype),
                hidden_states=(torch.ones(2, 3, 7, dtype=dtype),),
            )

    model = Model()
    converter = operations.ConvertOutputsToFp32(model.forward)
    expected = original(converter)
    actual = converter()
    assert actual.logits.dtype == expected.logits.dtype == torch.float32
    assert actual.hidden_states[0].dtype == expected.hidden_states[0].dtype == torch.float32
    assert not is_preserved_logits(actual.logits)
    # Even a manually supplied scope must not preserve CPU half tensors.
    token = _CONTEXT.set(_Scope(model, (2, 3), torch.device("cpu")))
    try:
        assert converter().logits.dtype == torch.float32
    finally:
        _CONTEXT.reset(token)


def test_scope_resets_on_exception_and_unknown_version_is_not_patched(monkeypatch):
    from solution import output_dtype

    model = torch.nn.Identity()
    trainer = SimpleNamespace(model=model)
    prior = object()
    token = output_dtype._CONTEXT.set(prior)
    try:
        with pytest.raises(RuntimeError, match="forward failed"):
            with output_dtype.training_logits_scope(trainer, model, {}):
                assert output_dtype._CONTEXT.get() is None
                raise RuntimeError("forward failed")
        assert output_dtype._CONTEXT.get() is prior
    finally:
        output_dtype._CONTEXT.reset(token)
    operations = pytest.importorskip("accelerate.utils.operations")
    current = operations.ConvertOutputsToFp32.__call__
    monkeypatch.setattr(output_dtype, "version", lambda package: "1.15.0")
    assert not output_dtype.patch_output_dtype()
    assert operations.ConvertOutputsToFp32.__call__ is current


@CUDA
@pytest.mark.parametrize("use_rewind", [False, True])
def test_real_accelerator_sft_and_rewind_preserve_values_gradients_rng_and_metrics(monkeypatch, use_rewind):
    accelerate = pytest.importorskip("accelerate")
    trl, rewind = _training_patches(monkeypatch)
    operations, original, wrapper = _conversion_patch(monkeypatch)
    from solution.output_dtype import _CONTEXT, is_preserved_logits
    from solution.test_padded_mlp import _unpatched_model_classes, _make_quantized_lora_model

    torch.manual_seed(211)
    accelerator = accelerate.Accelerator(mixed_precision="fp16")
    assert accelerator.device.type == "cuda" and accelerator.native_amp
    # Use the task's actual NF4 dispatch. Plain nn.Linear LoRA construction
    # probes unrelated optional PEFT quantizers (including installed torchao),
    # which is not the path this challenge trains and can fail before forward.
    modeling = _unpatched_model_classes(monkeypatch, v3=True)
    model, _, _ = _make_quantized_lora_model(modeling)
    model = accelerator.prepare_model(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    cls = rewind.make_rewind_trainer_class(trl.SFTTrainer) if use_rewind else trl.SFTTrainer
    initial_rng = _rng()

    def run(conversion):
        monkeypatch.setattr(operations.ConvertOutputsToFp32, "__call__", conversion)
        _restore_rng(initial_rng)
        trainer = _trainer(cls, model, accelerator)
        records = []
        if use_rewind:
            state = rewind.attach_rewind_state(
                trainer, SimpleNamespace(record_batch=lambda **record: records.append(record))
            )
            state.on_index(23)
            state.on_index(41)
        loss, outputs = trainer.compute_loss(
            model, _batch(), return_outputs=True, num_items_in_batch=torch.tensor(11, device="cuda")
        )
        # The randomly initialized tiny NF4 fixture can overflow at a large
        # synthetic scale even in the unmodified reference. Use the same
        # nontrivial scale as the NF4 padding/RoPE integration tests.
        gradients = torch.autograd.grad(loss * 128, parameters)
        if use_rewind:
            assert not state.failed and state.dropped == 0
        return loss.detach(), outputs, gradients, _rng(), trainer, records

    expected_loss, expected, expected_grad, expected_rng, reference, expected_records = run(original)
    actual_loss, actual, gradients, actual_rng, trainer, records = run(wrapper)
    assert expected.logits.dtype == torch.float32 and actual.logits.dtype == torch.float16
    assert is_preserved_logits(actual.logits) and not is_preserved_logits(expected.logits)
    assert actual.keys() == expected.keys()
    assert actual.loss.dtype == expected.loss.dtype == torch.float32
    torch.testing.assert_close(actual.logits.float(), expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for index, (gradient, reference_gradient) in enumerate(zip(gradients, expected_grad)):
        assert torch.isfinite(reference_gradient).all(), f"reference adapter gradient {index} is nonfinite"
        assert torch.isfinite(gradient).all(), f"optimized adapter gradient {index} is nonfinite"
        torch.testing.assert_close(gradient, reference_gradient, rtol=0, atol=0)
    assert torch.equal(actual_rng[0], expected_rng[0]) and torch.equal(actual_rng[1], expected_rng[1])
    assert dict(trainer._metrics["train"]) == dict(reference._metrics["train"])
    assert trainer._total_train_tokens == reference._total_train_tokens
    assert records == expected_records
    assert _CONTEXT.get() is None

    # A normal model call and eval go back through the full fp32 conversion.
    assert model(**_batch(), use_cache=False).logits.dtype == torch.float32
    model.eval()
    eval_trainer = _trainer(cls, model, accelerator)
    with torch.no_grad():
        _, eval_outputs = eval_trainer.compute_loss(model, _batch(), return_outputs=True)
    assert eval_outputs.logits.dtype == torch.float32


@CUDA
def test_marked_half_metrics_use_fp32_arithmetic_but_native_half_calls_keep_dtype(monkeypatch):
    trl, rewind = _training_patches(monkeypatch)
    from solution.output_dtype import _LOGITS_MARKER
    from solution import vocab

    torch.manual_seed(212)
    logits = torch.randn(2, 5, 4097, device="cuda", dtype=torch.float16)
    labels = torch.tensor([[-100, 0, 1, -100, 4096], [-100, -100, -100, -100, -100]], device="cuda")
    native_half = logits.clone()
    setattr(logits, _LOGITS_MARKER, True)
    with torch.no_grad():
        expected_entropy = trl.entropy_from_logits(logits.float())
        actual_entropy = trl.entropy_from_logits(logits)
        assert actual_entropy.dtype == expected_entropy.dtype == torch.float32
        torch.testing.assert_close(actual_entropy, expected_entropy, rtol=0, atol=0)
        assert trl.entropy_from_logits(native_half).dtype == torch.float16
        assert not vocab.supports_entropy(native_half)
    actual_loss, actual_counts = rewind.row_losses(logits, labels)
    expected_loss, expected_counts = rewind.row_losses(logits.float(), labels)
    assert actual_loss == expected_loss and actual_counts == expected_counts == [3, 0]
    native_reference = rewind.row_losses.__wrapped__(native_half, labels)
    assert rewind.row_losses(native_half, labels) == native_reference


@CUDA
@pytest.mark.parametrize("case", ["no_loss", "wrong_shape", "bf16", "other_output"])
def test_scoped_unsupported_outputs_convert_once_and_keep_other_fields(monkeypatch, case):
    operations, _, _ = _conversion_patch(monkeypatch)
    from solution.output_dtype import _CONTEXT, _Scope, is_preserved_logits
    from transformers.modeling_outputs import CausalLMOutputWithPast

    calls = []

    class Model(torch.nn.Module):
        def forward(self):
            calls.append(1)
            output = CausalLMOutputWithPast(
                loss=None if case == "no_loss" else torch.ones((), device="cuda", requires_grad=True),
                logits=torch.ones((2, 4 if case == "wrong_shape" else 3, 5), device="cuda",
                                  dtype=torch.bfloat16 if case == "bf16" else torch.float16),
                hidden_states=(torch.ones(2, 3, 7, device="cuda", dtype=torch.float16),),
            )
            return dict(output) if case == "other_output" else output

    model = Model()
    converter = operations.ConvertOutputsToFp32(model.forward)
    token = _CONTEXT.set(_Scope(model, (2, 3), torch.device("cuda", torch.cuda.current_device())))
    try:
        result = converter()
    finally:
        _CONTEXT.reset(token)
    assert calls == [1]
    assert result["logits"].dtype == torch.float32 and not is_preserved_logits(result["logits"])
    assert result["hidden_states"][0].dtype == torch.float32


@CUDA
def test_real_accelerate_training_avoids_full_fp32_output_allocation(monkeypatch):
    accelerate = pytest.importorskip("accelerate")
    trl, _ = _training_patches(monkeypatch)
    operations, original, wrapper = _conversion_patch(monkeypatch)
    accelerator = accelerate.Accelerator(mixed_precision="fp16")
    model = accelerator.prepare_model(_model(vocab=16384))
    batch = _batch(vocab=16384, batch=4, length=128)

    def run(conversion):
        monkeypatch.setattr(operations.ConvertOutputsToFp32, "__call__", conversion)
        trainer = _trainer(trl.SFTTrainer, model, accelerator)
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss, outputs = trainer.compute_loss(model, dict(batch), return_outputs=True)
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - start
        return extra, outputs.logits.dtype

    # Compile both input-storage specializations outside the measurement.
    run(original)
    run(wrapper)
    expected_peak, expected_dtype = run(original)
    actual_peak, actual_dtype = run(wrapper)
    assert expected_dtype == torch.float32 and actual_dtype == torch.float16
    fp32_bytes = 4 * 128 * 16384 * 4
    assert expected_peak - actual_peak >= fp32_bytes * 0.8
