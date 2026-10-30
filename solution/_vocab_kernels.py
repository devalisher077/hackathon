"""Original tiled vocabulary reductions and cross-entropy input gradients.

The vocabulary is split across programs so the 151,936-class task does not
require one enormous block. All partial statistics and reductions use fp32.
"""

import triton
import triton.language as tl


@triton.jit
def vocab_tile_stats_kernel(
    logits_ptr, maxima_ptr, sums_ptr, moments_ptr, row_stride,
    VOCAB: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr,
    WITH_ENTROPY: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    columns = tile * BLOCK + tl.arange(0, BLOCK)
    valid = columns < VOCAB
    values = tl.load(logits_ptr + row * row_stride + columns, mask=valid, other=-float("inf"))
    maximum = tl.max(values, axis=0)
    # Padded lanes contribute neither mass nor moment; avoid 0 * -inf there.
    shifted = tl.where(valid, values - maximum, 0.0)
    mass = tl.where(valid, tl.exp(shifted), 0.0)
    slot = row * TILES + tile
    tl.store(maxima_ptr + slot, maximum)
    tl.store(sums_ptr + slot, tl.sum(mass, axis=0))
    if WITH_ENTROPY:
        tl.store(moments_ptr + slot, tl.sum(mass * shifted, axis=0))


@triton.jit
def vocab_ce_reduce_kernel(
    logits_ptr, targets_ptr, maxima_ptr, sums_ptr,
    losses_ptr, row_maxima_ptr, inverse_sums_ptr, row_stride,
    VOCAB: tl.constexpr, TILES: tl.constexpr,
    IGNORE_INDEX: tl.constexpr, TILE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, TILE_BLOCK)
    valid = tile < TILES
    maxima = tl.load(maxima_ptr + row * TILES + tile, mask=valid, other=-float("inf"))
    sums = tl.load(sums_ptr + row * TILES + tile, mask=valid, other=0.0)
    maximum = tl.max(maxima, axis=0)
    shift = tl.where(valid, maxima - maximum, 0.0)
    scale = tl.where(valid, tl.exp(shift), 0.0)
    total = tl.sum(sums * scale, axis=0)

    target = tl.load(targets_ptr + row)
    selected = (target != IGNORE_INDEX) & (target >= 0) & (target < VOCAB)
    # Target validity is also checked before the launch. This mask ensures an
    # ignored or invalid target can never cause an out-of-bounds logits load.
    target_value = tl.load(logits_ptr + row * row_stride + target, mask=selected, other=0.0)
    loss = (maximum - target_value) + tl.log(total)
    tl.store(losses_ptr + row, tl.where(target == IGNORE_INDEX, 0.0, loss))
    tl.store(row_maxima_ptr + row, maximum)
    tl.store(inverse_sums_ptr + row, 1.0 / total)


@triton.jit
def vocab_ce_backward_kernel(
    logits_ptr, targets_ptr, row_maxima_ptr, inverse_sums_ptr,
    upstream_ptr, gradient_ptr, row_stride,
    VOCAB: tl.constexpr, IGNORE_INDEX: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    columns = tile * BLOCK + tl.arange(0, BLOCK)
    valid = columns < VOCAB
    values = tl.load(logits_ptr + row * row_stride + columns, mask=valid, other=0.0)
    maximum = tl.load(row_maxima_ptr + row)
    inverse_sum = tl.load(inverse_sums_ptr + row)
    target = tl.load(targets_ptr + row)
    upstream = tl.load(upstream_ptr + row)
    probability = tl.exp(values - maximum) * inverse_sum
    derivative = probability - (columns == target).to(tl.float32)
    # In particular, ignored rows have zero gradient even if an all-ignored
    # mean loss supplies an infinite upstream gradient (PyTorch's behavior).
    gradient = tl.where(target == IGNORE_INDEX, 0.0, derivative * upstream)
    tl.store(gradient_ptr + row * VOCAB + columns, gradient, mask=valid)


@triton.jit
def vocab_entropy_reduce_kernel(
    maxima_ptr, sums_ptr, moments_ptr, entropy_ptr,
    TILES: tl.constexpr, TILE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, TILE_BLOCK)
    valid = tile < TILES
    maxima = tl.load(maxima_ptr + row * TILES + tile, mask=valid, other=-float("inf"))
    sums = tl.load(sums_ptr + row * TILES + tile, mask=valid, other=0.0)
    moments = tl.load(moments_ptr + row * TILES + tile, mask=valid, other=0.0)
    maximum = tl.max(maxima, axis=0)
    shift = tl.where(valid, maxima - maximum, 0.0)
    scale = tl.where(valid, tl.exp(shift), 0.0)
    total = tl.sum(sums * scale, axis=0)
    centered_moment = tl.sum((moments + sums * shift) * scale, axis=0)
    # H = log(sum exp(x-M)) - sum(exp(x-M)*(x-M)) / sum exp(x-M).
    # The centering avoids cancellation from a large common logit offset.
    entropy = tl.log(total) - centered_moment / total
    tl.store(entropy_ptr + row, entropy)
