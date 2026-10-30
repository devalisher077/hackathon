"""Optional dense cache for frozen NF4 attention base projections.

Only Qwen2 attention's plain bitsandbytes base layers are wrapped. PEFT's
trainable branches, dropout, module calls and full token shapes stay unchanged.
The original bnb shape heuristic must select dequantization plus F.linear;
shapes using a fused quantized GEMM keep their original implementation.
"""

from dataclasses import dataclass
from functools import lru_cache, wraps
from importlib import import_module

import torch
from torch.autograd.function import once_differentiable

from .nf4 import _tensor_signature


_MARKER = "_soup_attention_dense_cache_forward"
_SETUP_MARKER = "_soup_attention_dense_cache_setup"
_CACHE_ATTR = "_soup_attention_dense_weight"
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class _Cache:
    signature: tuple
    sources: tuple
    weight: torch.Tensor


@lru_cache(maxsize=1)
def _backend():
    bnb = import_module("bitsandbytes")
    ops = import_module("bitsandbytes.backends.cuda.ops")
    return bnb, ops


def _signature(layer):
    weight, state = layer.weight, layer.weight.quant_state
    nested = state.state2
    tensors = weight, state.absmax, state.offset, nested.absmax, nested.code
    stream = torch.cuda.current_stream(weight.device).cuda_stream if weight.is_cuda else None
    return (
        id(weight), id(state), id(nested), stream, tuple(state.shape), state.dtype,
        state.blocksize, state.quant_type, state.nested, nested.blocksize,
        nested.dtype, nested.nested,
        tuple((id(value), _tensor_signature(value)) for value in tensors),
    )


def _cached_weight(layer, dequantize):
    signature = _signature(layer)
    state = layer.weight.quant_state
    nested = state.state2
    sources = (layer.weight, state, nested, state.absmax, state.offset, nested.absmax, nested.code)
    cached = getattr(layer, _CACHE_ATTR, None)
    if (cached is None or cached.signature != signature
            or any(old is not new for old, new in zip(cached.sources, sources))):
        with torch.no_grad():
            dense = dequantize(layer.weight, layer.weight.quant_state)
        # These are already owned by the layer. Keeping their identities alive
        # also prevents allocator/Python-id reuse from looking like a cache hit
        # after a caller replaces quantization state or parameter storage.
        cached = _Cache(signature, sources, dense)
        setattr(layer, _CACHE_ATTR, cached)
    return cached.weight


def _constant(value, device, dtype, count):
    return (isinstance(value, torch.Tensor) and value.device == device and value.dtype == dtype
            and not value.requires_grad and value.is_contiguous() and value.numel() == count)


def _supported(layer, x, bnb, ops):
    if (not isinstance(x, torch.Tensor) or not x.is_cuda or x.ndim not in (2, 3)
            or x.dtype not in (torch.float16, torch.float32) or not x.is_contiguous()
            or x.numel() == 0 or not torch.is_grad_enabled()
            or not layer.training):
        return False
    if (bnb.__version__ != "0.50.2" or type(layer) is not bnb.nn.Linear4bit
            or layer.compute_dtype != torch.float16 or layer.compute_type_is_set is not True
            or x.shape[-1] != layer.in_features or hasattr(layer, "_hf_hook")
            or any(parameter.requires_grad for parameter in layer.parameters())
            or (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != torch.float16)):
        return False
    count = layer.in_features * layer.out_features
    weight = layer.weight
    state = getattr(weight, "quant_state", None)
    nested = getattr(state, "state2", None)
    if (not _constant(weight, x.device, torch.uint8, (count + 1) // 2)
            or weight.ndim != 2 or weight.shape[1] != 1
            or getattr(weight, "bnb_quantized", None) is not True
            or getattr(state, "shape", None) != torch.Size((layer.out_features, layer.in_features))
            or getattr(state, "quant_type", None) != "nf4"
            or getattr(state, "dtype", None) != torch.float16
            or getattr(state, "blocksize", None) != 64 or count % 64
            or getattr(state, "nested", None) is not True
            or getattr(nested, "blocksize", None) != 256
            or getattr(nested, "dtype", None) != torch.float32
            or getattr(nested, "nested", None) is not False):
        return False
    blocks = count // 64
    if not (_constant(state.absmax, x.device, torch.uint8, blocks)
            and _constant(state.offset, x.device, torch.float32, 1)
            and _constant(nested.absmax, x.device, torch.float32, (blocks + 255) // 256)
            and _constant(nested.code, x.device, torch.float32, 256)):
        return False
    bias = layer.bias
    if bias is not None and not (
        isinstance(bias, torch.Tensor) and bias.device == x.device and not bias.requires_grad
        and bias.shape == (layer.out_features,) and bias.dtype in (torch.float16, torch.float32)
    ):
        return False
    selector = getattr(ops, "_gemm_4bit_use_custom_fn", None)
    ceiling = getattr(ops, "_gemm_4bit_custom_max_m", None)
    if not callable(selector) or type(ceiling) is not int or layer.in_features % 64:
        return False
    rows = x.numel() // x.shape[-1]
    return rows > ceiling or not selector(x.device.index, torch.float16, rows,
                                         layer.out_features, layer.in_features)


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(weight)
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        (weight,) = ctx.saved_tensors
        # This is the same native input-gradient operation as bnb's frozen
        # MatMul4Bit. The half rounding boundary remains before the fp32 cast.
        return torch.matmul(gradient, weight), None, None


def _install_projection(layer, linear_class):
    if getattr(layer.forward, _MARKER, False):
        return False
    original = layer.forward
    method = getattr(original, "__func__", None)
    if (type(layer) is not linear_class or "forward" in layer.__dict__
            or method is not linear_class.forward
            or getattr(method, "__module__", None) != "bitsandbytes.nn.modules"
            or getattr(method, "__qualname__", None) != "Linear4bit.forward"
            or hasattr(method, "__wrapped__") or hasattr(layer, "_hf_hook")):
        return False

    @wraps(original)
    def forward(x, *args, **kwargs):
        bnb, ops = _backend()
        if args or kwargs or not _supported(layer, x, bnb, ops):
            # In particular, moving a module away from CUDA should release its
            # stale device cache instead of silently retaining GPU storage.
            setattr(layer, _CACHE_ATTR, None)
            return original(x, *args, **kwargs)
        try:
            weight = _cached_weight(layer, bnb.functional.dequantize_4bit)
        except RuntimeError as error:
            if "Inference tensors do not track version counter" not in str(error):
                raise
            setattr(layer, _CACHE_ATTR, None)
            return original(x, *args, **kwargs)
        bias = layer.bias
        if bias is not None and bias.dtype != torch.float16:
            # Linear4bit makes this same persistent conversion on first use.
            bias.data = bias.data.to(torch.float16)
        return _FrozenLinear.apply(x.to(torch.float16), weight, bias).to(x.dtype)

    setattr(forward, _MARKER, True)
    layer.forward = forward
    return True


def cache_attention_weights(model, projections=_PROJECTIONS):
    """Install only on known CUDA Qwen2 attention base modules; return a count."""
    if not isinstance(model, torch.nn.Module) or getattr(getattr(model, "config", None), "model_type", None) != "qwen2":
        return 0
    if not any(parameter.is_cuda for parameter in model.parameters()):
        return 0
    if not projections or any(name not in _PROJECTIONS for name in projections):
        raise ValueError("attention cache accepts q_proj, k_proj, v_proj and o_proj")
    bnb, _ = _backend()
    if bnb.__version__ != "0.50.2":
        return 0
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    installed = 0
    for attention in model.modules():
        if type(attention) is not Qwen2Attention:
            continue
        for name in projections:
            projection = getattr(attention, name)
            base = getattr(projection, "base_layer", projection)
            installed += _install_projection(base, bnb.nn.Linear4bit)
    return installed


def patch_attention_cache(projections=_PROJECTIONS):
    """Chain installation after the existing Soup setup; no model settings change."""
    module = import_module("soup_cli.trainer.sft")
    wrapper_class = module.SFTTrainerWrapper
    original = wrapper_class._setup_transformers
    if getattr(original, _SETUP_MARKER, False):
        return

    @wraps(original)
    def setup(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        cache_attention_weights(getattr(self, "model", None), projections)
        return result

    setattr(setup, _SETUP_MARKER, True)
    wrapper_class._setup_transformers = setup
