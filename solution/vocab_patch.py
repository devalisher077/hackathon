"""Narrow vocabulary-operation patches for the challenge's pinned stack.

The trainer still owns causal alignment, metric masks, token accounting and the
rewind recorder. Only its numerical CE/entropy operations are replaced. The
reference contracts are Transformers 5.16.1, TRL 0.29.1 and Soup 0.77.0.
"""

from functools import wraps
from importlib import import_module

import torch

from . import vocab


_MARKER = "_soup_vocab_patch"


def _patch_fixed_cross_entropy(module):
    original = getattr(module, "fixed_cross_entropy", None)
    if not callable(original) or getattr(original, _MARKER, False):
        return

    @wraps(original)
    def fixed_cross_entropy(
        source, target, num_items_in_batch=None, ignore_index=-100, **kwargs
    ):
        if (
            not isinstance(source, torch.Tensor)
            or not isinstance(target, torch.Tensor)
            or not isinstance(ignore_index, int)
            or not vocab.supports_cross_entropy(source, target)
        ):
            return original(
                source, target, num_items_in_batch, ignore_index, **kwargs
            )

        token_loss = vocab.per_token_cross_entropy(source, target, ignore_index)
        total = token_loss.sum()
        if num_items_in_batch is not None:
            # Match fixed_cross_entropy: the trainer's denominator is not
            # necessarily the number of valid labels in this micro-batch.
            denominator = num_items_in_batch
            if torch.is_tensor(denominator):
                denominator = denominator.to(total.device)
        else:
            # Preserve mean CE, including NaN loss for an all-ignored batch.
            denominator = (target != ignore_index).sum()
        return total / denominator

    setattr(fixed_cross_entropy, _MARKER, True)
    module.fixed_cross_entropy = fixed_cross_entropy


def _patch_entropy(module):
    original = getattr(module, "entropy_from_logits", None)
    if not callable(original) or getattr(original, _MARKER, False):
        return

    @wraps(original)
    def entropy_from_logits(logits, chunk_size=128):
        if (
            torch.is_grad_enabled()
            or not isinstance(logits, torch.Tensor)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
            or not vocab.supports_entropy(logits)
        ):
            return original(logits, chunk_size=chunk_size)
        return vocab.entropy(logits)

    setattr(entropy_from_logits, _MARKER, True)
    # SFTTrainer imports this name from utils. Replacing only utils' attribute
    # would leave its already-bound reference unchanged.
    module.entropy_from_logits = entropy_from_logits


def _supports_rewind(logits, labels):
    return (
        isinstance(logits, torch.Tensor)
        and isinstance(labels, torch.Tensor)
        and logits.ndim == 3
        and labels.ndim == 2
        and logits.shape[:2] == labels.shape
        and logits.shape[0] > 0
        and logits.shape[1] > 1
        and vocab.supports_cross_entropy(logits[0, :-1, :], labels[0, 1:])
    )


def _patch_rewind(module):
    original = getattr(module, "row_losses", None)
    if not callable(original) or getattr(original, _MARKER, False):
        return

    @wraps(original)
    def row_losses(logits, labels):
        if not _supports_rewind(logits, labels):
            return original(logits, labels)

        losses = []
        counts = []
        # Retain Soup's per-example loop and its fp32 accumulation/counting.
        # No logits upcast or concatenation across examples is introduced.
        with torch.no_grad():
            for row in range(labels.shape[0]):
                targets = labels[row, 1:]
                token_loss = vocab.per_token_cross_entropy(
                    logits[row, :-1, :].detach(), targets, ignore_index=-100
                ).float()
                mask = (targets != -100).to(token_loss.dtype)
                count = mask.sum()
                total = (token_loss * mask).sum()
                losses.append(float(total / count.clamp(min=1.0)))
                counts.append(int(count.item()))
        return losses, counts

    setattr(row_losses, _MARKER, True)
    module.row_losses = row_losses


def _optional_module(name):
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        # Some Soup regression tests replace optional frameworks with stubs.
        # Skip an absent target module, but surface missing dependencies inside
        # a real module rather than silently suppressing a broken installation.
        if exc.name and (name == exc.name or name.startswith(exc.name + ".")):
            return None
        raise


def patch_vocab_ops():
    """Install idempotent patches; unsupported calls keep their original path."""
    targets = (
        ("transformers.loss.loss_utils", _patch_fixed_cross_entropy),
        ("trl.trainer.sft_trainer", _patch_entropy),
        ("soup_cli.trainer.rewind_hf", _patch_rewind),
    )
    for name, patch in targets:
        module = _optional_module(name)
        if module is not None:
            patch(module)
