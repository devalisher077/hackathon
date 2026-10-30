"""Keep losslessly represented half logits inside the fixed training metrics.

Accelerate normally widens every output tensor after its AMP forward. The
training loss already retains the original half logits; widening them only for
entropy, accuracy and rewind duplicates a large vocabulary matrix. Those three
consumers can read the half storage with the same fp32 arithmetic instead.

The scope is established only by our guarded SFT loss path. Ordinary model
calls, evaluation, other models and unsupported outputs keep Accelerate's
original conversion. Other output fields still receive that conversion.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import torch


_MARKER = "_soup_scoped_half_logits"
_LOGITS_MARKER = "_soup_fp32_metrics_half_storage"
_CONTEXT = ContextVar("soup_training_logits_storage", default=None)


@dataclass(frozen=True)
class _Scope:
    model: torch.nn.Module
    shape: tuple
    device: torch.device


def is_preserved_logits(logits):
    """Whether this tensor represents an otherwise-fp32 metric input."""
    return (
        isinstance(logits, torch.Tensor)
        and logits.is_cuda
        and logits.dtype == torch.float16
        and getattr(logits, _LOGITS_MARKER, False) is True
    )


def _scope_for(trainer, model, inputs):
    args = getattr(trainer, "args", None)
    accelerator = getattr(trainer, "accelerator", None)
    labels = inputs.get("labels") if isinstance(inputs, dict) else None
    if (
        not isinstance(model, torch.nn.Module)
        or not model.training
        or not getattr(getattr(trainer, "model", None), "training", False)
        or not torch.is_grad_enabled()
        or getattr(args, "fp16", None) is not True
        or getattr(args, "bf16", None) is not False
        or getattr(accelerator, "native_amp", None) is not True
        or getattr(accelerator, "mixed_precision", None) != "fp16"
        or not isinstance(labels, torch.Tensor)
        or not labels.is_cuda
        or labels.ndim != 2
        or labels.dtype != torch.int64
        or getattr(getattr(model, "config", None), "model_type", None) != "qwen2"
    ):
        return None

    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    if type(model) is not Qwen2ForCausalLM:
        from peft import PeftModelForCausalLM

        if type(model) is not PeftModelForCausalLM or type(model.get_base_model()) is not Qwen2ForCausalLM:
            return None
    return _Scope(model, tuple(labels.shape), labels.device)


@contextmanager
def training_logits_scope(trainer, model, inputs):
    """Always restore the caller's scope, including exceptions and nesting."""
    token = _CONTEXT.set(_scope_for(trainer, model, inputs))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def _matches_model(converter, args, state):
    # Accelerator.prepare_model normally wraps an unbound method, with the
    # prepared model as args[0]. Also support its documented bound-call path.
    if args and args[0] is state.model:
        return True
    forward = converter.model_forward
    return (
        getattr(forward, "__self__", None) is state.model
        or getattr(getattr(forward, "__wrapped__", None), "__self__", None) is state.model
    )


def patch_output_dtype():
    """Install the pinned Accelerate conversion specialization once."""
    try:
        if version("accelerate") != "1.14.0":
            return False
        operations = import_module("accelerate.utils.operations")
    except (PackageNotFoundError, ModuleNotFoundError):
        return False
    converter_class = getattr(operations, "ConvertOutputsToFp32", None)
    original = getattr(converter_class, "__call__", None)
    if getattr(original, _MARKER, False):
        return True
    if (
        getattr(original, "__module__", None) != "accelerate.utils.operations"
        or getattr(original, "__qualname__", None) != "ConvertOutputsToFp32.__call__"
        or hasattr(original, "__wrapped__")
    ):
        return False

    @wraps(original)
    def convert(self, *args, **kwargs):
        state = _CONTEXT.get()
        if state is None or not _matches_model(self, args, state):
            return original(self, *args, **kwargs)

        # The AMP-decorated callable and its RNG consumption are unchanged.
        # If its result is unsupported, convert that result; never run it twice.
        outputs = self.model_forward(*args, **kwargs)
        from transformers.modeling_outputs import CausalLMOutputWithPast

        logits = getattr(outputs, "logits", None)
        loss = getattr(outputs, "loss", None)
        if (
            type(outputs) is not CausalLMOutputWithPast
            or not isinstance(logits, torch.Tensor)
            or not logits.is_cuda
            or logits.device != state.device
            or logits.dtype != torch.float16
            or logits.ndim != 3
            or tuple(logits.shape[:2]) != state.shape
            or not 0 < logits.shape[-1] <= 262144
            or not logits.is_contiguous()
            or not isinstance(loss, torch.Tensor)
            or loss.device != state.device
            or loss.dtype != torch.float32
            or loss.numel() != 1
            or not loss.requires_grad
        ):
            return operations.convert_to_fp32(outputs)

        setattr(logits, _LOGITS_MARKER, True)
        return type(outputs)(**{
            key: value if key == "logits" else operations.convert_to_fp32(value)
            for key, value in outputs.items()
        })

    setattr(convert, _MARKER, True)
    converter_class.__call__ = convert
    return True
