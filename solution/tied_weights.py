"""Lossless half storage for the task's frozen, tied embedding/head weight.

Soup loads this frozen base in fp16 on a T4, then PEFT widens non-quantized
parameters to fp32. We verify every weight value before reclaiming that storage.
Embedding outputs retain fp32; only the already-fp16 AMP head path uses half
storage directly. All other head calls reconstruct the original fp32 weight.
"""

from functools import wraps
from importlib import import_module

import torch
from torch.nn import functional as F


_MARKER = "_soup_lossless_tied_half"
_SETUP_MARKER = "_soup_tied_weights_setup"
_CHECK_CHUNK = 1 << 22  # At most 16 MiB for each temporary fp32 round-trip.


def _lossless_half_copy(weight):
    """Return a half copy only if every finite fp32 value survives exactly.

Checks stay on the tensor's device; one final scalar synchronization decides
whether to install the representation. This routine never consumes RNG state.
"""
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.float32
        or not weight.is_contiguous()
        or weight.numel() == 0
    ):
        return None
    with torch.no_grad():
        half = weight.to(dtype=torch.float16)
        source = weight.detach().view(-1)
        packed = half.view(-1)
        checks = []
        for start in range(0, source.numel(), _CHECK_CHUNK):
            stop = min(start + _CHECK_CHUNK, source.numel())
            values = source[start:stop]
            round_trip = packed[start:stop].float()
            checks.append(((round_trip == values) & torch.isfinite(values)).all())
        if not torch.stack(checks).all().item():
            return None
        return half


def _pair(model):
    if not isinstance(model, torch.nn.Module):
        return None
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen2":
        return None
    get_embedding = getattr(model, "get_input_embeddings", None)
    get_head = getattr(model, "get_output_embeddings", None)
    if not callable(get_embedding) or not callable(get_head):
        return None
    embedding, head = get_embedding(), get_head()
    if type(embedding) is not torch.nn.Embedding or type(head) is not torch.nn.Linear:
        return None
    if (
        embedding.weight is not head.weight
        or type(embedding.weight) is not torch.nn.Parameter
        or embedding.weight.requires_grad
        or head.bias is not None
        or embedding.max_norm is not None
        or not embedding.weight.is_cuda
        or not embedding.weight.is_contiguous()
        or embedding.weight.shape != (embedding.num_embeddings, embedding.embedding_dim)
        or (head.out_features, head.in_features) != tuple(embedding.weight.shape)
    ):
        return None
    return embedding, head


def compress_tied_weights(model):
    """Install the guarded representation, returning whether it is active.

The Parameter object, its names/shapes, sharing and requires_grad are retained.
No trainable parameters, buffers, model settings or training options are added.
"""
    pair = _pair(model)
    if pair is None:
        return False
    embedding, head = pair
    if (
        getattr(embedding.forward, _MARKER, False)
        and getattr(head.forward, _MARKER, False)
        and embedding.weight.dtype == torch.float16
    ):
        return True
    # A custom forward could use the parameter differently; do not bypass it.
    if (
        embedding.weight.dtype != torch.float32
        or getattr(embedding.forward, "__func__", None) is not torch.nn.Embedding.forward
        or getattr(head.forward, "__func__", None) is not torch.nn.Linear.forward
    ):
        return False

    half = _lossless_half_copy(embedding.weight)
    if half is None:
        return False
    original_embedding = embedding.forward
    original_head = head.forward

    @wraps(original_embedding)
    def embedding_forward(input):
        # Casting inside forward means existing module post-hooks observe the
        # same fp32 output, including hooks that enable input gradients.
        return original_embedding(input).float()

    @wraps(original_head)
    def head_forward(input):
        if (
            isinstance(input, torch.Tensor)
            and input.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
        ):
            return original_head(input)
        # With AMP disabled (or another AMP dtype), retain the original fp32
        # operand and let F.linear obey the caller's unchanged autocast state.
        return F.linear(input, head.weight.float(), head.bias)

    setattr(embedding_forward, _MARKER, True)
    setattr(head_forward, _MARKER, True)
    with torch.no_grad():
        embedding.weight.data = half
    embedding.forward = embedding_forward
    head.forward = head_forward
    return True


def patch_tied_weights():
    """Run after Soup finishes model/PEFT setup, before creating its trainer."""
    module = import_module("soup_cli.trainer.sft")
    wrapper_class = module.SFTTrainerWrapper
    original = wrapper_class._setup_transformers
    if getattr(original, _SETUP_MARKER, False):
        return

    @wraps(original)
    def setup(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        compress_tied_weights(getattr(self, "model", None))
        return result

    setattr(setup, _SETUP_MARKER, True)
    wrapper_class._setup_transformers = setup
