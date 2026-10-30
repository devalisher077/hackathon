"""Skip right-padding positions inside the task's frozen, tokenwise MLPs.

Attention and its q/v LoRA calls retain their full original shapes, masks and
dropout RNG sequence. Only plain frozen MLP projections receive a compact
token matrix; their output is scattered back before the residual addition.
No collator, batch, position, training setting or metric is changed.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import inspect

import torch

from .scatter import ScatterRows


_PADDING_CONTEXT = ContextVar("soup_mlp_padding_context", default=None)
_MARKER = "_soup_padding_mlp_patch"


@dataclass(frozen=True)
class _PaddingState:
    batch: int
    sequence: int
    device: torch.device
    indices: torch.Tensor
    mlp_ids: frozenset
    attention_ids: frozenset = frozenset()


def _model_padding_state(model, arguments):
    """Build one selection for an ordinary, uncached right-padded forward."""
    if (
        not model.training
        or not torch.is_grad_enabled()
        or getattr(model, "gradient_checkpointing", False)
        or arguments.get("past_key_values") is not None
    ):
        return None
    extras = arguments.get("kwargs", {})
    if not isinstance(extras, dict):
        return None
    use_cache = arguments.get("use_cache")
    if use_cache is None:
        use_cache = getattr(model.config, "use_cache", True)
    if (
        use_cache is not False
        or extras.get("cache_position") is not None
        or extras.get("output_attentions", getattr(model.config, "output_attentions", False))
        or extras.get("output_hidden_states", getattr(model.config, "output_hidden_states", False))
    ):
        return None

    input_ids, embeds = arguments.get("input_ids"), arguments.get("inputs_embeds")
    if (input_ids is None) == (embeds is None):
        return None
    source = input_ids if input_ids is not None else embeds
    expected_rank = 2 if input_ids is not None else 3
    mask = arguments.get("attention_mask")
    if (
        not isinstance(source, torch.Tensor)
        or not source.is_cuda
        or source.ndim != expected_rank
        or not isinstance(mask, torch.Tensor)
        or not mask.is_cuda
        or mask.device != source.device
        or mask.ndim != 2
        or tuple(mask.shape) != tuple(source.shape[:2])
        or mask.shape[0] == 0
        or mask.shape[1] == 0
        or mask.dtype not in (
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
            torch.int64, torch.float16, torch.float32, torch.float64,
        )
    ):
        return None

    # Left/interior padding is deliberately unsupported: the logit before a
    # first valid label could belong to a masked position and still contribute
    # to causal loss. Nonempty 1...1,0...0 rows avoid that ambiguity.
    binary = ((mask == 0) | (mask == 1)).all()
    right_padded = (mask[:, 1:] <= mask[:, :-1]).all()
    nonempty_rows = (mask[:, 0] == 1).all()
    if not bool((binary & right_padded & nonempty_rows).item()):
        return None
    indices = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
    if indices.numel() == mask.numel():
        return None
    mlp_ids = frozenset(id(layer.mlp) for layer in model.layers)
    attention_ids = frozenset(
        id(getattr(projection, "base_layer", projection))
        for layer in model.layers
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        for projection in (getattr(layer.self_attn, name),)
    )
    return _PaddingState(mask.shape[0], mask.shape[1], source.device, indices, mlp_ids, attention_ids)


def _supported_mlp(module, x, state, mlp_class):
    if (
        state is None
        or type(module) is not mlp_class
        or id(module) not in state.mlp_ids
        or not module.training
        or not torch.is_grad_enabled()
        or not isinstance(x, torch.Tensor)
        or not x.is_cuda
        or x.device != state.device
        or x.dtype not in (torch.float16, torch.float32)
        or x.ndim != 3
        or tuple(x.shape[:2]) != (state.batch, state.sequence)
        or x.shape[-1] != module.hidden_size
        or not x.is_contiguous()
        or not x.requires_grad
        or getattr(module.config, "hidden_act", None) != "silu"
    ):
        return False

    from bitsandbytes.nn import Linear4bit

    for projection in (module.gate_proj, module.up_proj, module.down_proj):
        # A PEFT wrapper can contain dropout even when its parameters happen
        # to be frozen. Exact plain projection types exclude that RNG change.
        if type(projection) is not Linear4bit:
            return False
        weight = projection.weight
        quant_state = getattr(weight, "quant_state", None)
        if (
            weight.requires_grad
            or not weight.is_cuda
            or weight.device != x.device
            or weight.dtype != torch.uint8
            or projection.bias is not None
            or projection.compute_dtype != torch.float16
            or projection.compute_type_is_set is not True
            or quant_state is None
            or getattr(quant_state, "quant_type", None) != "nf4"
            or any(parameter.requires_grad for parameter in projection.parameters())
        ):
            return False
    return True


def patch_padding_mlp():
    """Wrap Qwen2's model and the currently installed MLP implementation once."""
    from transformers.models.qwen2 import modeling_qwen2

    model_class, mlp_class = modeling_qwen2.Qwen2Model, modeling_qwen2.Qwen2MLP
    original_model = model_class.forward
    original_mlp = mlp_class.forward
    if getattr(original_model, _MARKER, False) or getattr(original_mlp, _MARKER, False):
        return
    try:
        signature = inspect.signature(original_model)
    except (TypeError, ValueError):
        signature = None
    expected = {"input_ids", "inputs_embeds", "attention_mask", "past_key_values", "use_cache"}
    known_signature = signature is not None and expected.issubset(signature.parameters)

    @wraps(original_model)
    def model_forward(self, *args, **kwargs):
        state = None
        if type(self) is model_class and known_signature:
            try:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                # Let the original callable report its own argument error.
                bound = None
            if bound is not None:
                state = _model_padding_state(self, bound.arguments)
        token = _PADDING_CONTEXT.set(state)
        try:
            return original_model(self, *args, **kwargs)
        finally:
            _PADDING_CONTEXT.reset(token)

    @wraps(original_mlp)
    def mlp_forward(self, x, *args, **kwargs):
        state = _PADDING_CONTEXT.get()
        if args or kwargs or not _supported_mlp(self, x, state, mlp_class):
            return original_mlp(self, x, *args, **kwargs)
        flat = x.view(-1, x.shape[-1])
        compact = flat.index_select(0, state.indices)
        # Reuse the v3 dtype optimization and unchanged example SwiGLU path;
        # no projection implementation or rounding boundary is replaced here.
        compact_output = original_mlp(self, compact)
        output = ScatterRows.apply(compact_output, state.indices, flat.shape[0])
        return output.view(state.batch, state.sequence, compact_output.shape[-1])

    setattr(model_forward, _MARKER, True)
    setattr(mlp_forward, _MARKER, True)
    model_class.forward = model_forward
    mlp_class.forward = mlp_forward
