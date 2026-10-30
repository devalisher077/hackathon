"""Parallel warmup contracts and real CUDA cache reuse, without timing thresholds."""

from types import SimpleNamespace

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_plan_uses_observed_widths_quantization_and_target_alignment():
    from solution.compile_kernels import _Pointer, _build_warmups

    work = _build_warmups([(96, 1e-5), (96, 1e-5)],
                         [(torch.float16, 128, 12288)], 4101, [160, 160])
    norms = [item for item in work if item.module == "_rmsnorm_kernels"]
    assert len(norms) == 2
    assert all(item.kwargs["WIDTH"] == 96 and item.kwargs["BLOCK"] == 128 for item in norms)
    assert norms[0].kwargs["EPSILON"] == 1e-5
    quant = [item for item in work if item.module == "_nf4_kernels"]
    assert len(quant) == 1
    assert quant[0].kwargs["QUANT_BLOCK"] == 128 and quant[0].args[-1] == 12288
    assert quant[0].args[3] == _Pointer(torch.float16)
    reductions = [item for item in work if item.name == "vocab_ce_reduce_kernel"]
    assert {item.args[0].dtype for item in reductions} == {torch.float16, torch.float32}
    assert {item.args[1].offset for item in reductions} == {0, 8}
    assert all(item.kwargs["VOCAB"] == 4101 and item.kwargs["TILES"] == 2
               and item.kwargs["TILE_BLOCK"] == 2 and item.args[-1] == 4101
               for item in reductions)
    assert len([item for item in work if item.module == "swiglu"]) == 2


def test_cpu_and_unknown_models_do_not_import_gpu_compiler(monkeypatch):
    import solution.compile_kernels as implementation

    def unexpected(*args):
        raise AssertionError("unsupported model must not compile")

    monkeypatch.setattr(implementation, "_run_warmups", unexpected)
    assert implementation.prepare_model_kernels(None) == 0
    model = torch.nn.Linear(7, 11)
    model.config = SimpleNamespace(model_type="qwen2")
    before = model.weight.detach().clone()
    state = torch.get_rng_state().clone()
    assert implementation.prepare_model_kernels(model) == 0
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), state)


def test_optional_rope_plan_derives_dimensions_and_runtime_length_classes():
    from solution.compile_kernels import _build_warmups

    assert _build_warmups([], [], None, []) == []
    work = _build_warmups([], [], None, [], rope=(6, 2, 64))
    assert len(work) == 4
    assert {item.kwargs["T"] for item in work} == {127, 128}
    assert all(item.kwargs["D"] == 64 and item.kwargs["Q"] == 384
               and item.kwargs["K"] == 128 and item.kwargs["BLOCK"] == 512
               and item.kwargs["COS_BROADCAST"] is True for item in work)


def test_half_training_plan_only_prepares_half_vocab_signatures():
    from solution.compile_kernels import _build_warmups

    assert _build_warmups([], [], None, []) == []
    work = _build_warmups([], [], 151936, [], vocab_dtypes=(torch.float16,))
    vocab = [item for item in work if item.name in (
        "vocab_tile_stats_kernel", "vocab_ce_reduce_kernel", "vocab_ce_backward_kernel"
    )]
    assert vocab and all(item.args[0].dtype == torch.float16 for item in vocab)


def test_training_plan_omits_strided_backward_and_warms_one_fused_signature():
    from solution.compile_kernels import _Pointer, _build_warmups

    work = _build_warmups([], [], 151936, [], fused_widths=(1536, 1536), strided_ce_backward=False)
    reductions = [item for item in work if item.name == "vocab_ce_reduce_kernel"]
    backwards = [item for item in work if item.name == "vocab_ce_backward_kernel"]
    assert {item.args[1].offset for item in reductions} == {0, 8}
    assert {item.args[1].offset for item in backwards} == {0}
    fused = [item for item in work if item.name == "mlp_input_grad_sum_kernel"]
    assert len(fused) == 1
    assert fused[0].args == (_Pointer(torch.float16), _Pointer(torch.float16), _Pointer(torch.float32), 1536)
    assert fused[0].kwargs == {"BLOCK": 1024, "num_warps": 4, "enable_fp_fusion": False}


def test_setup_chains_existing_wrapper_and_preserves_return_and_errors(monkeypatch):
    import solution.compile_kernels as implementation

    events = []
    result = object()
    model = object()

    class Wrapper:
        def _setup_transformers(self, *args, **kwargs):
            events.append(("original", args, kwargs))
            if kwargs.get("fail"):
                raise RuntimeError("setup failed")
            self.model = model
            return result

    monkeypatch.setattr(implementation, "import_module",
                        lambda name: SimpleNamespace(SFTTrainerWrapper=Wrapper))
    monkeypatch.setattr(implementation, "prepare_model_kernels",
                        lambda value: events.append(("compile", value)))
    original = Wrapper._setup_transformers
    implementation.patch_parallel_compile()
    installed = Wrapper._setup_transformers
    implementation.patch_parallel_compile()
    assert Wrapper._setup_transformers is installed and installed.__wrapped__ is original
    assert Wrapper()._setup_transformers(1, dtype="half") is result
    assert events == [("original", (1,), {"dtype": "half"}), ("compile", model)]
    with pytest.raises(RuntimeError, match="setup failed"):
        Wrapper()._setup_transformers(fail=True)
    assert events[-1] == ("original", (), {"fail": True})


def test_handle_preparation_deduplicates_serializes_sources_and_selects_device(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Barrier, get_ident, local
    import solution.compile_kernels as implementation

    selected = local()
    barrier = Barrier(2)
    events = []
    device = torch.device("cuda", 3)

    @contextmanager
    def device_scope(value):
        assert value == device
        selected.device = value
        try:
            yield
        finally:
            del selected.device

    monkeypatch.setattr(torch.cuda, "device", device_scope)

    class Kernel:
        def __init__(self, source, name):
            self.src = SimpleNamespace(fn=source)
            self.name, self.initializations, self._run = name, 0, None

        def _init_handles(self):
            assert selected.device == device
            self.initializations += 1
            events.append((self.name, get_ident()))
            if self.name in ("a0", "b0"):
                # Both source groups must progress simultaneously; variants
                # within either group must still be visited in input order.
                barrier.wait(timeout=10)
            self._run = lambda *args: pytest.fail("a launcher must never be invoked")

        @property
        def run(self):
            if self._run is None:
                self._init_handles()
            return self._run

    a, b, c = object(), object(), object()
    a0, a1, b0, b1, c0 = (Kernel(source, name) for source, name in
                           ((a, "a0"), (a, "a1"), (b, "b0"), (b, "b1"), (c, "c0")))
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert implementation._initialize_handles(
            device, [a0, b0, a1, a0, None, b1, c0, b0], executor
        ) == 5
    assert all(kernel.initializations == 1 for kernel in (a0, a1, b0, b1, c0))
    for prefix in ("a", "b"):
        group = [(name, thread) for name, thread in events if name.startswith(prefix)]
        assert [name for name, _ in group] == [prefix + "0", prefix + "1"]
        assert len({thread for _, thread in group}) == 1
    worker_ids = {thread for _, thread in events}
    assert len(worker_ids) == 2 and get_ident() not in worker_ids


def test_warmups_finalize_futures_before_preparing_handles(monkeypatch):
    from contextlib import nullcontext
    from contextvars import ContextVar
    import sys
    import solution.compile_kernels as implementation

    active_mode = ContextVar("test_compile_mode", default=None)
    events = []
    compiled = SimpleNamespace(src=SimpleNamespace(fn=object()))

    class FutureKernel:
        ready = False

        def result(self):
            assert self.ready and active_mode.get() is None
            events.append("resolved")
            return compiled

    future_kernel = FutureKernel()

    class Mode:
        def __init__(self, executor):
            self.executor = executor

        def __enter__(self):
            active_mode.set(self)
            return self

        def __exit__(self, *args):
            future_kernel.ready = True
            active_mode.set(None)
            events.append("finalized")

    async_compile = SimpleNamespace(active_mode=active_mode, AsyncCompileMode=Mode,
                                    FutureKernel=FutureKernel)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="3.6.0"))
    monkeypatch.setitem(sys.modules, "triton.runtime", SimpleNamespace(_async_compile=async_compile))
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", SimpleNamespace(MockTensor=object))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    results = iter((future_kernel, compiled, None))

    def warmup(*args, **kwargs):
        assert active_mode.get() is not None
        events.append("warmup")
        return next(results)

    monkeypatch.setattr(implementation, "import_module",
                        lambda name: SimpleNamespace(kernel=SimpleNamespace(warmup=warmup)))

    def initialize(device, kernels, executor):
        assert active_mode.get() is None
        assert kernels == [compiled, compiled, None]
        events.append("handles")

    monkeypatch.setattr(implementation, "_initialize_handles", initialize)
    work = [implementation._Warmup("fake", "kernel", (), {})] * 3
    assert implementation._run_warmups(torch.device("cuda", 2), work) == 3
    assert events == ["warmup"] * 3 + ["finalized", "resolved", "handles"]
    assert active_mode.get() is None


def test_handle_worker_error_is_propagated_after_other_worker_finishes(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from threading import Barrier, Event
    import solution.compile_kernels as implementation

    barrier, finished = Barrier(2), Event()
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    class Kernel:
        def __init__(self, fail):
            self.src, self.fail = SimpleNamespace(fn=object()), fail

        @property
        def run(self):
            barrier.wait(timeout=10)
            if self.fail:
                raise RuntimeError("native handle failed")
            finished.set()
            return lambda *args: pytest.fail("a launcher must never be invoked")

    with pytest.raises(RuntimeError, match="native handle failed"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            implementation._initialize_handles(torch.device("cuda", 0),
                                               [Kernel(True), Kernel(False)], executor)
    assert finished.is_set()


@CUDA
def test_cuda_warmup_allocates_no_model_tensors_and_reuses_real_launch_signatures(monkeypatch):
    triton = pytest.importorskip("triton")
    if triton.__version__ != "3.6.0":
        pytest.skip("the optional compiler preparation supports pinned Triton 3.6.0")
    import solution.compile_kernels as implementation
    from solution.compile_kernels import _build_warmups, _run_warmups
    from solution.nf4 import _launch_nf4
    from solution.rmsnorm import rmsnorm
    from solution.swiglu import swiglu
    from solution.vocab import _tile_statistics, entropy, per_token_cross_entropy_fp32
    from solution import _nf4_kernels, _rmsnorm_kernels, _vocab_kernels, _rope_kernels, _mlp_grad_kernels
    from solution.rope import _RotaryPair
    import solution.swiglu as swiglu_module

    torch.manual_seed(987)
    device = torch.device("cuda", torch.cuda.current_device())
    # Non-power-of-two dimensions exercise the same derived constexpr logic
    # without allocating the task's full vocabulary or loading a checkpoint.
    width, intermediate, vocab = 96, 160, 4101
    work = _build_warmups([(width, 1e-5)], [(torch.float16, 128, 12288)],
                         vocab, [intermediate], rope=(3, 2, 32), fused_widths=(width,))
    prepared = []
    original_initialize = implementation._initialize_handles

    def observe_handles(device, kernels, executor):
        result = original_initialize(device, kernels, executor)
        prepared.extend(kernel for kernel in kernels if kernel is not None)
        return result

    monkeypatch.setattr(implementation, "_initialize_handles", observe_handles)
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone()
    allocated = torch.cuda.memory_allocated()
    assert _run_warmups(device, work) == len(work)
    assert torch.cuda.memory_allocated() == allocated
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
    assert prepared and all(kernel.module is not None and kernel.function is not None
                            and kernel._run is not None for kernel in prepared)

    # Any unmatched dtype, alignment, integer or constexpr key would call the
    # real compiler again. Observing _do_compile distinguishes real cache reuse
    # from simply moving compilation to a different point in the test.
    misses = []
    modules = (_nf4_kernels, _rmsnorm_kernels, _vocab_kernels, _rope_kernels, _mlp_grad_kernels, swiglu_module)
    seen = set()
    for module in modules:
        for kernel in vars(module).values():
            if hasattr(kernel, "_do_compile") and id(kernel) not in seen:
                seen.add(id(kernel))
                original = kernel._do_compile

                def observed(*args, _original=original, _kernel=kernel, **kwargs):
                    misses.append(_kernel.fn.__name__)
                    return _original(*args, **kwargs)

                monkeypatch.setattr(kernel, "_do_compile", observed)

    x = torch.randn(6, width, device=device, requires_grad=True)
    weights = torch.ones(width, device=device)
    rmsnorm(x, weights, eps=1e-5).sum().backward()
    gate = torch.randn(6, intermediate, device=device, dtype=torch.float16, requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    swiglu(gate, up).sum().backward()
    gate_dx = torch.randn(6, width, device=device, dtype=torch.float16)
    up_dx = torch.randn_like(gate_dx)
    dx = torch.empty(gate_dx.shape, device=device, dtype=torch.float32)
    _mlp_grad_kernels.mlp_input_grad_sum_kernel[(1,)](
        gate_dx, up_dx, dx, dx.numel(), BLOCK=1024, num_warps=4, enable_fp_fusion=False
    )
    torch.testing.assert_close(dx, gate_dx.float() + up_dx.float(), rtol=0, atol=0)
    packed = torch.zeros(6144, device=device, dtype=torch.uint8)
    scales = torch.ones(96, device=device)
    _launch_nf4(packed, scales, 128, torch.empty(12288, device=device, dtype=torch.float16))
    for time in (125, 128):
        q = torch.randn(2, time, 3, 32, device=device, requires_grad=True).transpose(1, 2)
        k = torch.randn(2, time, 2, 32, device=device, requires_grad=True).transpose(1, 2)
        cos, sin = (torch.randn(1, time, 32, device=device) for _ in range(2))
        oq, ok = _RotaryPair.apply(q, k, cos, sin)
        # Contiguous B,H,T,D gradient has different stride VALUES from the
        # forward's B,T,H,D storage, but the same specialization alignments.
        torch.autograd.grad((oq, ok), (q, k),
                            (torch.randn(oq.shape, device=device), torch.randn(ok.shape, device=device)))
    for dtype in (torch.float16, torch.float32):
        for offset in (0, 1):
            logits = torch.randn(3, vocab, device=device, dtype=dtype, requires_grad=True)
            targets = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int64)[offset:offset + 3]
            per_token_cross_entropy_fp32(logits, targets).sum().backward()
            with torch.no_grad():
                _tile_statistics(logits, with_entropy=True)
                if dtype == torch.float32:
                    entropy(logits)
    torch.cuda.synchronize()
    assert misses == []


@CUDA
def test_cuda_model_metadata_is_observed_without_changing_weights(monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    import solution.compile_kernels as implementation

    config = Qwen2Config(vocab_size=257, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, rms_norm_eps=1e-5)
    model = Qwen2ForCausalLM(config).cuda()
    mlp = model.model.layers[0].mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        original = getattr(mlp, name)
        linear = bnb.nn.Linear4bit(original.in_features, original.out_features, bias=False,
                                   compute_dtype=torch.float16, quant_type="nf4")
        linear.weight = bnb.nn.Params4bit(original.weight.detach().half().cpu(),
                                          requires_grad=False, quant_type="nf4")
        setattr(mlp, name, linear.cuda())
    model.requires_grad_(False)
    metadata = [(id(p), p.data_ptr(), p.dtype, tuple(p.shape)) for p in model.parameters()]
    calls = []
    monkeypatch.setattr(implementation, "_run_warmups",
                        lambda device, work: calls.append((device, work)) or len(work))
    assert implementation.prepare_model_kernels(model) > 0
    assert metadata == [(id(p), p.data_ptr(), p.dtype, tuple(p.shape)) for p in model.parameters()]
    work = calls[0][1]
    assert {item.kwargs["WIDTH"] for item in work if "WIDTH" in item.kwargs} == {64}
    assert {item.kwargs["VOCAB"] for item in work if "VOCAB" in item.kwargs} == {257}
    assert {item.kwargs["QUANT_BLOCK"] for item in work if "QUANT_BLOCK" in item.kwargs}
