"""Compare the small metrics specialization with the pinned, real TRL method."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _install(monkeypatch):
    module = pytest.importorskip("trl.trainer.sft_trainer")
    from solution.metrics import patch_sft_metrics

    original = module.SFTTrainer.compute_loss
    while getattr(original, "_soup_no_logits_metric_copy", False):
        original = original.__wrapped__
    # monkeypatch restores the incoming state even if apply() ran first.
    monkeypatch.setattr(module.SFTTrainer, "compute_loss", original)
    assert patch_sft_metrics()
    wrapper = module.SFTTrainer.compute_loss
    assert wrapper is not original
    assert patch_sft_metrics() and module.SFTTrainer.compute_loss is wrapper
    return module, original


class _Gather:
    num_processes = 2
    parallelism_config = None

    def __init__(self):
        self.calls = []

    def gather_for_metrics(self, value):
        self.calls.append(value.detach().clone())
        # Exercise exactly the same distributed mean/sum semantics as TRL.
        return torch.stack((value, value))


class _SingleGather(_Gather):
    num_processes = 1

    def gather_for_metrics(self, value):
        self.calls.append(value.detach().clone())
        return value


def test_scalar_transfer_preserves_mixed_types_and_large_integer_counts():
    from solution.metrics import _scalars_to_host

    values = (torch.tensor(0.125), torch.tensor(2**62 + 73),
              torch.tensor(-0.0, dtype=torch.float64), torch.tensor(True))
    actual = _scalars_to_host(*values)
    expected = [value.item() for value in values]
    assert actual == expected
    assert [type(value) for value in actual] == [float, int, float, bool]
    assert torch.signbit(torch.tensor(actual[2]))


class _SmallLM(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 8, device=device)
        self.dropout = torch.nn.Dropout(0.25)
        self.head = torch.nn.Linear(8, 17, device=device)
        self.calls = []

    def forward(self, input_ids, labels, attention_mask, use_cache, num_items_in_batch=None):
        from transformers.modeling_outputs import CausalLMOutput

        self.calls.append((use_cache, num_items_in_batch))
        logits = self.head(self.dropout(self.embedding(input_ids)))
        token_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 17), labels[:, 1:].reshape(-1),
            reduction="sum", ignore_index=-100,
        )
        denominator = (labels[:, 1:] != -100).sum().clamp(min=1)
        if num_items_in_batch is not None:
            denominator = num_items_in_batch
        self.last_output = CausalLMOutput(loss=token_loss / denominator, logits=logits)
        return self.last_output


def _trainer(cls, model):
    # Do not download a model/tokenizer or build a dataloader for compute_loss.
    # The actual Transformers Trainer.compute_loss is used, with its required
    # state supplied explicitly, rather than replacing the superclass method.
    trainer = object.__new__(cls)
    trainer.model = model
    trainer.args = SimpleNamespace(
        use_liger_kernel=False, loss_type="nll", average_tokens_across_devices=False, n_gpu=1,
    )
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = False
    trainer.padding_free = False
    trainer.compute_loss_func = None
    trainer.label_smoother = None
    trainer.model_accepts_loss_kwargs = True
    trainer.accelerator = _Gather()
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._total_train_tokens = 37
    trainer.state = SimpleNamespace(global_step=4)
    return trainer


def _inputs(device, ignored=False):
    ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], device=device)
    labels = torch.tensor([[-100, 1, 2, -100, -100], [-100, 6, 7, 8, -100]], device=device)
    if ignored:
        labels.fill_(-100)
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], device=device),
        "_prediction_loss_only": False,
    }


def _compare_metrics(left, right):
    assert dict(left._metrics["train"]) == dict(right._metrics["train"])
    assert dict(left._metrics["eval"]) == dict(right._metrics["eval"])
    assert left._total_train_tokens == right._total_train_tokens
    assert len(left.accelerator.calls) == len(right.accelerator.calls)
    for actual, expected in zip(left.accelerator.calls, right.accelerator.calls):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("training, return_outputs", [(True, True), (False, False)])
def test_real_trainer_loss_outputs_gradients_rng_and_metrics(monkeypatch, device, training, return_outputs):
    module, original = _install(monkeypatch)
    torch.manual_seed(104)
    model = _SmallLM(device).train(training)
    reference_model = _SmallLM(device).train(training)
    reference_model.load_state_dict(model.state_dict())
    trainer = _trainer(module.SFTTrainer, model)
    reference = _trainer(module.SFTTrainer, reference_model)
    inputs, ref_inputs = _inputs(device), _inputs(device)
    denominator = torch.tensor(11, device=device)

    torch.manual_seed(105)
    expected = original(reference, reference_model, ref_inputs, return_outputs, denominator)
    expected_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    torch.manual_seed(105)
    actual = trainer.compute_loss(model, inputs, return_outputs, denominator)
    actual_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    assert torch.equal(actual_rng, expected_rng)
    if return_outputs:
        actual_loss, output = actual
        expected_loss, ref_output = expected
        assert output is model.last_output and ref_output is reference_model.last_output
        torch.testing.assert_close(output.logits, ref_output.logits, rtol=0, atol=0)
    else:
        actual_loss, expected_loss = actual, expected
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    actual_loss.mul(128).backward()
    expected_loss.mul(128).backward()
    for parameter, ref_parameter in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=0, atol=0)
    _compare_metrics(trainer, reference)
    assert model.calls == reference_model.calls == [(False, denominator)]
    assert set(inputs) == set(ref_inputs) and "_prediction_loss_only" not in inputs
    assert inputs["use_cache"] is ref_inputs["use_cache"] is False
    assert trainer._total_train_tokens == (51 if training else 37)


def test_all_ignored_labels_preserve_zero_accuracy(monkeypatch):
    module, original = _install(monkeypatch)
    model = _SmallLM("cpu").eval()
    trainer, reference = _trainer(module.SFTTrainer, model), _trainer(module.SFTTrainer, model)
    expected = original(reference, model, _inputs("cpu", ignored=True))
    actual = trainer.compute_loss(model, _inputs("cpu", ignored=True))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _compare_metrics(trainer, reference)
    assert trainer._metrics["eval"]["mean_token_accuracy"] == [0.0]


@CUDA
@pytest.mark.parametrize("training, ignored", [(True, False), (True, True), (False, False), (False, True)])
def test_single_process_batches_metric_transfer_with_exact_values(monkeypatch, training, ignored):
    from solution import metrics

    module, original = _install(monkeypatch)
    torch.manual_seed(191)
    model = _SmallLM("cuda").train(training)
    reference_model = _SmallLM("cuda").train(training)
    reference_model.load_state_dict(model.state_dict())
    trainer = _trainer(module.SFTTrainer, model)
    reference = _trainer(module.SFTTrainer, reference_model)
    trainer.accelerator = _SingleGather()
    reference.accelerator = _SingleGather()
    transfers = []
    transfer = metrics._scalars_to_host

    def observed(*values):
        transfers.append(len(values))
        return transfer(*values)

    monkeypatch.setattr(metrics, "_scalars_to_host", observed)
    initial_rng = torch.cuda.get_rng_state()
    expected, expected_output = original(reference, reference_model, _inputs("cuda", ignored), True)
    expected_rng = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(initial_rng)
    actual, output = trainer.compute_loss(model, _inputs("cuda", ignored), True)
    assert transfers == [3 if training else 2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(output.logits, expected_output.logits, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    _compare_metrics(trainer, reference)
    assert type(trainer._total_train_tokens) is int
    assert trainer._total_train_tokens == (44 if training else 37)
    if ignored:
        assert trainer._metrics["train" if training else "eval"]["mean_token_accuracy"] == [0.0]


@pytest.mark.parametrize(
    "change", ["liger", "dft", "virtual", "aux", "padding_free", "custom_loss", "smoother", "shift", "no_mask"],
)
def test_unsupported_modes_keep_original_path(monkeypatch, change):
    module = pytest.importorskip("trl.trainer.sft_trainer")
    from solution.metrics import patch_sft_metrics

    calls = []
    sentinel = object()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        calls.append((self, model, inputs, return_outputs, num_items_in_batch))
        return sentinel

    # A signature-compatible stand-in makes any original fallback observable.
    compute_loss.__module__ = "trl.trainer.sft_trainer"
    compute_loss.__qualname__ = "SFTTrainer.compute_loss"
    monkeypatch.setattr(module.SFTTrainer, "compute_loss", compute_loss)
    assert patch_sft_metrics()
    model = _SmallLM("cpu")
    trainer, inputs = _trainer(module.SFTTrainer, model), _inputs("cpu")
    if change == "liger":
        trainer.args.use_liger_kernel = True
    elif change == "dft":
        trainer.args.loss_type = "dft"
    elif change == "virtual":
        trainer.num_virtual_tokens = 4
    elif change == "aux":
        trainer.aux_loss_enabled = True
    elif change == "padding_free":
        trainer.padding_free = True
    elif change == "custom_loss":
        trainer.compute_loss_func = lambda *args: None
    elif change == "smoother":
        trainer.label_smoother = object()
    elif change == "shift":
        inputs["shift_labels"] = inputs["labels"]
    else:
        inputs.pop("attention_mask")
    assert trainer.compute_loss(model, inputs, True, 123) is sentinel
    assert calls == [(trainer, model, inputs, True, 123)]
    assert "_prediction_loss_only" in inputs and "use_cache" not in inputs


def test_unknown_version_or_method_is_not_replaced(monkeypatch):
    module = pytest.importorskip("trl.trainer.sft_trainer")
    from solution import metrics

    current = module.SFTTrainer.compute_loss
    monkeypatch.setattr(metrics, "version", lambda package: "0.29.2")
    assert not metrics.patch_sft_metrics() and module.SFTTrainer.compute_loss is current
    monkeypatch.setattr(metrics, "version", lambda package: "0.29.1")
    unknown = lambda *args, **kwargs: None
    monkeypatch.setattr(module.SFTTrainer, "compute_loss", unknown)
    assert not metrics.patch_sft_metrics() and module.SFTTrainer.compute_loss is unknown


def test_real_soup_rewind_subclass_keeps_records_and_full_outputs(monkeypatch):
    module, original = _install(monkeypatch)
    rewind = pytest.importorskip("soup_cli.trainer.rewind_hf")
    rewind_cls = rewind.make_rewind_trainer_class(module.SFTTrainer)
    model = _SmallLM("cpu").train()
    records = []
    sink = SimpleNamespace(record_batch=lambda **record: records.append(record))
    trainer = _trainer(rewind_cls, model)
    state = rewind.attach_rewind_state(trainer, sink)
    state.on_index(23)
    state.on_index(41)
    loss, outputs = trainer.compute_loss(model, _inputs("cpu"), return_outputs=True)
    expected_losses, expected_counts = rewind.row_losses(outputs.logits, _inputs("cpu")["labels"])
    assert outputs is model.last_output and loss is outputs.loss
    assert not state.failed and state.dropped == 0
    assert records == [{"step": 5, "micro": 0, "rows": [23, 41], "row_loss": expected_losses,
                        "row_tokens": expected_counts}]
    assert trainer._metrics["train"]["num_tokens"] == [51]


@CUDA
def test_cuda_accuracy_drops_vocabulary_copy_allocation(monkeypatch):
    module, original = _install(monkeypatch)
    # Isolate the accuracy allocation: entropy is tested above using the real
    # upstream helper; here its small result avoids masking the copy's peak.
    monkeypatch.setattr(module, "entropy_from_logits", lambda x: torch.zeros_like(x[..., 0]))
    shape = (4, 128, 8192)
    logits = torch.randn(shape, device="cuda")
    loss = torch.tensor(0.0, device="cuda")
    outputs = SimpleNamespace(logits=logits, loss=loss)
    base = module.SFTTrainer.__mro__[1]

    def base_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        assert return_outputs
        return loss, outputs

    monkeypatch.setattr(base, "compute_loss", base_loss)
    model = torch.nn.Identity().cuda().train()

    def measure(method):
        trainer = _trainer(module.SFTTrainer, model)
        inputs = {"labels": torch.zeros(shape[:2], device="cuda", dtype=torch.long),
                  "attention_mask": torch.ones(shape[:2], device="cuda", dtype=torch.long)}
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual_loss, actual_outputs = method(trainer, model, inputs, return_outputs=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - start
        assert actual_loss is loss and actual_outputs is outputs
        return peak, trainer

    expected_peak, reference = measure(original)
    actual_peak, trainer = measure(module.SFTTrainer.compute_loss)
    _compare_metrics(trainer, reference)
    copied_bytes = shape[0] * (shape[1] - 1) * shape[2] * logits.element_size()
    assert expected_peak - actual_peak >= copied_bytes * 0.9
