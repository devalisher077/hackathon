"""Avoid a vocabulary-sized copy in ordinary TRL 0.29.1 token accuracy.

The loss still comes from the next trainer in the original MRO. This small
specialization preserves the upstream metric reductions and full model outputs;
only argmax runs before the causal sequence slice. Unsupported configurations
and other TRL versions keep the original compute_loss implementation.
"""

from functools import wraps
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from inspect import Parameter, signature

import torch

from .output_dtype import training_logits_scope


_MARKER = "_soup_no_logits_metric_copy"


def _single_process_transfer(trainer, labels):
    processes = getattr(getattr(trainer, "accelerator", None), "num_processes", None)
    return (
        labels.is_cuda
        and type(processes) is int
        and processes == 1
        and getattr(trainer.args, "n_gpu", None) == 1
    )


def _scalars_to_host(*values):
    """One transfer preserving each scalar's original bits and Python type.

Packing bytes avoids narrowing integer token counts to a floating-point type.
Place wider scalar types first so every CPU reinterpretation stays aligned.
"""
    order = sorted(range(len(values)), key=lambda index: -values[index].element_size())
    packed = torch.cat([values[index].reshape(1).view(torch.uint8) for index in order]).cpu()
    results = [None] * len(values)
    offset = 0
    for index in order:
        width = values[index].element_size()
        results[index] = packed.narrow(0, offset, width).view(values[index].dtype).item()
        offset += width
    return results


def _supported(trainer, inputs):
    args = getattr(trainer, "args", None)
    if (
        type(inputs) is not dict
        or getattr(args, "use_liger_kernel", None) is not False
        or getattr(args, "loss_type", None) != "nll"
        or getattr(trainer, "num_virtual_tokens", None) != 0
        or getattr(trainer, "aux_loss_enabled", None) is not False
        or getattr(trainer, "padding_free", None) is not False
        or getattr(trainer, "compute_loss_func", None) is not None
        or getattr(trainer, "label_smoother", None) is not None
        or "shift_labels" in inputs
    ):
        return False
    labels = inputs.get("labels")
    mask = inputs.get("attention_mask")
    return (
        isinstance(labels, torch.Tensor)
        and isinstance(mask, torch.Tensor)
        and labels.ndim == 2
        and labels.shape == mask.shape
        and labels.dtype == torch.int64
        and labels.device == mask.device
    )


def _known_method(function):
    if (
        getattr(function, "__module__", None) != "trl.trainer.sft_trainer"
        or getattr(function, "__qualname__", None) != "SFTTrainer.compute_loss"
        or hasattr(function, "__wrapped__")
    ):
        return False
    try:
        parameters = list(signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    return (
        [parameter.name for parameter in parameters]
        == ["self", "model", "inputs", "return_outputs", "num_items_in_batch"]
        and all(parameter.kind == Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters)
        and all(parameter.default is Parameter.empty for parameter in parameters[:3])
        and parameters[3].default is False
        and parameters[4].default is None
    )


def patch_sft_metrics():
    """Install the version-guarded specialization, returning whether active."""
    try:
        if version("trl") != "0.29.1":
            return False
        module = import_module("trl.trainer.sft_trainer")
    except (PackageNotFoundError, ModuleNotFoundError):
        return False
    trainer_class = getattr(module, "SFTTrainer", None)
    original = getattr(trainer_class, "compute_loss", None)
    if getattr(original, _MARKER, False):
        return True
    if not _known_method(original):
        return False

    @wraps(original)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _supported(self, inputs):
            return original(
                self, model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        labels = inputs["labels"]
        batch_transfer = _single_process_transfer(self, labels)
        inputs["use_cache"] = False
        # This is precisely the super target of the original SFTTrainer method,
        # including when Soup's outer RewindTrainer calls us via its own super.
        with training_logits_scope(self, model, inputs):
            loss, outputs = super(trainer_class, self).compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )

        with torch.no_grad():
            per_token_entropy = module.entropy_from_logits(outputs.logits)
            mask = inputs["attention_mask"]
            entropy = torch.sum(per_token_entropy * mask) / mask.sum()
            entropy = self.accelerator.gather_for_metrics(entropy).mean()
            if not batch_transfer:
                entropy = entropy.item()
        if mode == "train":
            tokens = self.accelerator.gather_for_metrics(mask.sum()).sum()
            if not batch_transfer:
                tokens = tokens.item()

        with torch.no_grad():
            # Reduce contiguous full rows, then discard the final prediction.
            # Slicing logits first and calling contiguous copies B*(T-1)*V
            # values when B>1; slicing these integer predictions is tiny.
            predictions = outputs.logits.argmax(dim=-1)[..., :-1]
            targets = labels[..., 1:].contiguous()
            supervised = targets != -100
            correct = ((predictions == targets) & supervised).sum()
            total = supervised.sum()
            correct = self.accelerator.gather_for_metrics(correct)
            total = self.accelerator.gather_for_metrics(total)
            total_sum = total.sum()
            if batch_transfer:
                quotient = correct.sum() / total_sum.clamp(min=1)
                accuracy = torch.where(total_sum > 0, quotient, torch.zeros_like(quotient))
                if mode == "train":
                    entropy, tokens, accuracy = _scalars_to_host(entropy, tokens, accuracy)
                else:
                    entropy, accuracy = _scalars_to_host(entropy, accuracy)
            else:
                accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["entropy"].append(entropy)
        if mode == "train":
            self._total_train_tokens += tokens
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)
        return (loss, outputs) if return_outputs else loss

    setattr(compute_loss, _MARKER, True)
    trainer_class.compute_loss = compute_loss
    return True
