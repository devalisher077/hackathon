"""Compile the model's ordinary kernel signatures concurrently during setup.

This uses Triton 3.6's own asynchronous compiler with two in-process workers.
Every future is joined before setup returns. Compilation stays inside the same
training command and its normal cache. Native launchers and binary handles are
also prepared without invoking a launcher. Preparation does not launch GPU work,
allocate model-sized tensors, consume random numbers, or change kernel calls.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
import math

import torch


_MARKER = "_soup_parallel_kernel_compile"


@dataclass(frozen=True)
class _Pointer:
    dtype: torch.dtype
    offset: int = 0


@dataclass
class _Warmup:
    module: str
    name: str
    args: tuple
    kwargs: dict


def _build_warmups(norms, quantized, vocab_size, intermediate_sizes, rope=None, fused_widths=(),
                   *, vocab_dtypes=(torch.float16, torch.float32), strided_ce_backward=True):
    """Describe signatures from actual module metadata, without allocating tensors."""
    f32, f16, i64, u8 = (_Pointer(dtype) for dtype in
                         (torch.float32, torch.float16, torch.int64, torch.uint8))
    work = []

    def add(module, name, args, **kwargs):
        candidate = _Warmup(module, name, tuple(args), kwargs)
        if candidate not in work:
            work.append(candidate)

    for width, epsilon in sorted(set(norms)):
        block = 1 << (width - 1).bit_length()
        common = dict(WIDTH=width, BLOCK=block, num_warps=4, enable_fp_fusion=False)
        add("_rmsnorm_kernels", "rmsnorm_forward_kernel", (f32,) * 4,
            EPSILON=epsilon, **common)
        add("_rmsnorm_kernels", "rmsnorm_input_grad_kernel", (f32,) * 5, **common)

    for dtype, blocksize, elements in quantized:
        add("_nf4_kernels", "nf4_unpack_scale_kernel",
            (u8, f32, f32, _Pointer(dtype), elements),
            QUANT_BLOCK=blocksize, TILE=1024, num_warps=4, enable_fp_fusion=False)

    # Row count is a runtime value. One representative multiple of the actual
    # intermediate width covers its ordinary integer-alignment specialization.
    for width in sorted(set(intermediate_sizes)):
        add("swiglu", "example_swiglu_fwd_kernel", (f16, f16, f16, width * 16), BLOCK=1024)
        add("swiglu", "example_swiglu_bwd_kernel", (f16,) * 5 + (width * 16,), BLOCK=1024)

    # The fused MLP sum has no shape constexpr. For the actual aligned hidden
    # widths, M*width has the same ordinary integer-alignment specialization
    # for every runtime token count. Other widths can compile lazily.
    for width in sorted(set(fused_widths)):
        if isinstance(width, int) and width > 0 and width % 16 == 0:
            add("_mlp_grad_kernels", "mlp_input_grad_sum_kernel", (f16, f16, f32, width),
                BLOCK=1024, num_warps=4, enable_fp_fusion=False)

    if rope is not None:
        heads, key_heads, width = rope
        qwidth, kwidth = heads * width, key_heads * width
        block = 1 << (qwidth + kwidth - 1).bit_length()
        common = dict(D=width, Q=qwidth, K=kwidth, COS_BROADCAST=True,
                      SIN_BROADCAST=True, BLOCK=block, num_warps=4, enable_fp_fusion=False)
        # Sequence length is a runtime integer, specialized only on alignment.
        # These cover ordinary divisible-by-16 and other sequence lengths.
        for time in (128, 127):
            add("_rope_kernels", "rope_pair_forward_kernel",
                (f32,) * 6 + (time * qwidth, qwidth, time * kwidth, kwidth,
                              time * width, width, time * width, width), T=time, **common)
            add("_rope_kernels", "rope_pair_backward_kernel",
                (f32,) * 6 + (time * qwidth, width, qwidth, 1,
                              time * kwidth, width, kwidth, 1,
                              time * width, width, time * width, width), T=time, **common)

    if vocab_size is not None:
        from .vocab import VOCAB_BLOCK

        tiles = (vocab_size + VOCAB_BLOCK - 1) // VOCAB_BLOCK
        tile_block = 1 << (tiles - 1).bit_length()
        common = dict(num_warps=4, enable_fp_fusion=False)
        for dtype in vocab_dtypes:
            logits = _Pointer(dtype)
            for with_entropy in (False, True):
                add("_vocab_kernels", "vocab_tile_stats_kernel",
                    (logits, f32, f32, f32, vocab_size),
                    VOCAB=vocab_size, TILES=tiles, BLOCK=VOCAB_BLOCK,
                    WITH_ENTROPY=with_entropy, **common)
            # Causal loss creates an aligned target buffer; rewind can use
            # labels[row, 1:] directly, whose int64 pointer is eight bytes off.
            for targets in (i64, _Pointer(torch.int64, 8)):
                add("_vocab_kernels", "vocab_ce_reduce_kernel",
                    (logits, targets, f32, f32, f32, f32, f32, vocab_size),
                    VOCAB=vocab_size, TILES=tiles, IGNORE_INDEX=-100,
                    TILE_BLOCK=tile_block, **common)
                if targets.offset == 0 or strided_ce_backward:
                    add("_vocab_kernels", "vocab_ce_backward_kernel",
                        (logits, targets, f32, f32, f32, logits, vocab_size),
                        VOCAB=vocab_size, IGNORE_INDEX=-100, BLOCK=VOCAB_BLOCK, **common)
        add("_vocab_kernels", "vocab_entropy_reduce_kernel", (f32,) * 4,
            TILES=tiles, TILE_BLOCK=tile_block, **common)
    return work


def _model_warmups(model):
    if not isinstance(model, torch.nn.Module):
        return None
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen2":
        return None
    parameters = list(model.parameters())
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        return None

    from bitsandbytes.nn import Linear4bit
    from transformers.models.qwen2 import modeling_qwen2
    Qwen2MLP, Qwen2RMSNorm = modeling_qwen2.Qwen2MLP, modeling_qwen2.Qwen2RMSNorm

    norms, quantized, intermediate_sizes, fused_widths = [], [], [], []
    fused_active = getattr(Qwen2MLP.forward, "_soup_fused_mlp_patch", False)
    for module in model.modules():
        if (type(module) is Qwen2RMSNorm and module.weight.dtype == torch.float32
                and not module.weight.requires_grad and 0 < module.weight.numel() <= 8192):
            norms.append((module.weight.numel(), float(module.variance_epsilon)))
        elif type(module) is Linear4bit:
            state = getattr(module.weight, "quant_state", None)
            if (state is not None and getattr(state, "quant_type", None) == "nf4"
                    and state.dtype in (torch.float16, torch.float32)
                    and state.blocksize in (32, 64, 128, 256, 512, 1024, 2048, 4096)):
                quantized.append((state.dtype, state.blocksize, math.prod(state.shape)))
        elif type(module) is Qwen2MLP and getattr(module.config, "hidden_act", None) == "silu":
            if all(type(layer) is Linear4bit and layer.compute_dtype == torch.float16
                   for layer in (module.gate_proj, module.up_proj, module.down_proj)):
                intermediate_sizes.append(module.intermediate_size)
                if fused_active and all(
                    layer.bias is None and not layer.weight.requires_grad
                    and getattr(getattr(layer.weight, "quant_state", None), "quant_type", None) == "nf4"
                    and getattr(getattr(layer.weight, "quant_state", None), "dtype", None) == torch.float16
                    for layer in (module.gate_proj, module.up_proj, module.down_proj)
                ):
                    fused_widths.append(module.hidden_size)

    head_getter = getattr(model, "get_output_embeddings", None)
    head = head_getter() if callable(head_getter) else None
    vocab_size = None
    vocab_dtypes = (torch.float16, torch.float32)
    if (type(head) is torch.nn.Linear and not head.weight.requires_grad
            and head.weight.dtype in (torch.float16, torch.float32)
            and 0 < head.out_features <= 262144):
        vocab_size = head.out_features
        if head.weight.dtype == torch.float16:
            # The fixed training path keeps AMP logits in half storage. An
            # unsupported/evaluation fp32 consumer can still compile lazily.
            vocab_dtypes = (torch.float16,)
    rope = None
    if getattr(modeling_qwen2.apply_rotary_pos_emb, "_soup_rope_pair_patch", False):
        heads = getattr(config, "num_attention_heads", 0)
        key_heads = getattr(config, "num_key_value_heads", 0)
        hidden_size = getattr(config, "hidden_size", 0)
        width = getattr(config, "head_dim", None) or (hidden_size // heads if heads else 0)
        if (isinstance(width, int) and isinstance(heads, int) and isinstance(key_heads, int)
                and heads > 0 and key_heads > 0 and 0 < width <= 256 and width % 2 == 0
                and (heads + key_heads) * width <= 8192):
            rope = heads, key_heads, width
    work = _build_warmups(
        norms, quantized, vocab_size, intermediate_sizes, rope, fused_widths,
        vocab_dtypes=vocab_dtypes, strided_ce_backward=False,
    )
    return next(iter(devices)), work


def _initialize_handles(device, kernels, executor):
    """Load unique compiled kernels; serialize variants of each source function.

    Triton 3.6's ``run`` property builds its C launcher and loads the binary.
    Reading the property does not invoke that launcher or allocate its scratch
    tensors. Same-source variants often share launcher C code, so keeping them
    together avoids concurrently compiling the same cache entry.
    """
    groups, seen = {}, set()
    for kernel in kernels:
        if kernel is None or id(kernel) in seen:
            continue
        seen.add(id(kernel))
        groups.setdefault(id(kernel.src.fn), []).append(kernel)

    def initialize_group(group):
        # CUDA's current device is thread-local. The compiled model's device
        # must be selected in each worker, including on multi-GPU hosts.
        with torch.cuda.device(device):
            for kernel in group:
                getattr(kernel, "run")

    futures = [executor.submit(initialize_group, group) for group in groups.values()]
    for future in futures:
        future.result()
    return len(seen)


def _run_warmups(device, work):
    import triton
    if triton.__version__ != "3.6.0":
        return 0
    from triton.runtime.jit import MockTensor
    from triton.runtime import _async_compile

    if _async_compile.active_mode.get() is not None:
        return 0

    class MockPointer(MockTensor):
        def __init__(self, pointer):
            super().__init__(pointer.dtype)
            self.offset = pointer.offset

        def data_ptr(self):
            return self.offset

    # Binding/signature collection stays on this thread. Triton dispatches only
    # compilation to the executor, then finalizes the cache entries on exit.
    with torch.cuda.device(device), ThreadPoolExecutor(max_workers=2) as executor:
        token = _async_compile.active_mode.set(None)
        pending = []
        try:
            with _async_compile.AsyncCompileMode(executor):
                for item in work:
                    kernel = getattr(import_module(f"solution.{item.module}"), item.name)
                    args = tuple(MockPointer(value) if isinstance(value, _Pointer) else value
                                 for value in item.args)
                    pending.append(kernel.warmup(*args, grid=(1,), **item.kwargs))
        finally:
            # Triton 3.6's context resets its variable after finalizing futures.
            # Restore it even if compilation fails before that final line.
            _async_compile.active_mode.reset(token)
        # AsyncCompileMode has finalized every compile on the calling thread.
        # Cache hits may already be CompiledKernel objects, while misses return
        # FutureKernel objects whose result is now ready.
        kernels = [result.result() if isinstance(result, _async_compile.FutureKernel) else result
                   for result in pending]
        _initialize_handles(device, kernels, executor)
    return len(work)


def prepare_model_kernels(model):
    """Warm up only recognized CUDA models; unsupported models remain lazy."""
    collected = _model_warmups(model)
    if collected is None:
        return 0
    device, work = collected
    return _run_warmups(device, work) if work else 0


def patch_parallel_compile():
    """Chain after the existing Soup setup, including its tied-weight wrapper."""
    module = import_module("soup_cli.trainer.sft")
    wrapper_class = module.SFTTrainerWrapper
    original = wrapper_class._setup_transformers
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def setup(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        prepare_model_kernels(getattr(self, "model", None))
        return result

    setattr(setup, _MARKER, True)
    wrapper_class._setup_transformers = setup
