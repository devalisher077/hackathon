"""NF4 format, backend integration and fallback checks.

Run on Kaggle: ``python -m pytest solution/test_nf4.py -v``.
The CPU fallback test does not validate the Triton kernel.
"""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def backend(monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    ops = import_module("bitsandbytes.backends.cuda.ops")
    original = ops._dequantize_4bit_impl
    if getattr(original, "_soup_custom_nf4", False):
        original = original.__wrapped__
    # Keep the numerical reference unpatched; restore the incoming state later.
    monkeypatch.setattr(ops, "_dequantize_4bit_impl", original)
    return bnb, ops, original


def _install_and_observe_launches(monkeypatch):
    import solution.nf4 as implementation

    calls = []
    launch = implementation._launch_nf4

    def observed(*args):
        calls.append(args[-1].numel())
        return launch(*args)

    monkeypatch.setattr(implementation, "_launch_nf4", observed)
    implementation.patch_nf4()
    return calls


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("nested", [False, True])
def test_public_dequant_matches_bnb_with_and_without_doublequant(backend, monkeypatch, dtype, nested):
    bnb, ops, original = backend
    torch.manual_seed(71)
    weight = torch.randn(17, 1536, device="cuda", dtype=dtype)
    packed, state = bnb.functional.quantize_4bit(
        weight, blocksize=64, compress_statistics=nested, quant_type="nf4",
    )
    expected = bnb.functional.dequantize_4bit(packed, state)
    calls = _install_and_observe_launches(monkeypatch)
    actual = bnb.functional.dequantize_4bit(packed, state)
    assert calls == [weight.numel()]
    assert actual.shape == weight.shape and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_helper_odd_length_reuses_output_and_returns_none(backend, monkeypatch, dtype):
    _, ops, original = backend
    # Construct packed bytes directly to cover the odd last nibble without
    # depending on whether a particular quantizer accepts odd-sized inputs.
    torch.manual_seed(72)
    elements, blocksize = 195, 64
    packed = torch.randint(0, 256, ((elements + 1) // 2, 1), device="cuda", dtype=torch.uint8)
    scales = torch.rand((elements + blocksize - 1) // blocksize, device="cuda") + 0.01
    expected = torch.empty((3, 65), device="cuda", dtype=dtype)
    original(packed, scales, blocksize, "nf4", dtype, out=expected)
    calls = _install_and_observe_launches(monkeypatch)
    output = torch.full_like(expected, float("nan"))
    pointer = output.data_ptr()
    result = ops._dequantize_4bit_impl(packed, scales, blocksize, "nf4", dtype, out=output)
    assert result is None and output.data_ptr() == pointer
    assert calls == [elements]
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)


@CUDA
def test_all_sixteen_nf4_codes_use_high_nibble_first(backend, monkeypatch):
    bnb, ops, _ = backend
    packed = torch.tensor([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF], device="cuda", dtype=torch.uint8)
    scales = torch.tensor([2.0], device="cuda")
    expected = bnb.functional.get_4bit_type("nf4", device="cuda") * scales[0]
    output = torch.empty(16, device="cuda")
    _install_and_observe_launches(monkeypatch)
    ops._dequantize_4bit_impl(packed, scales, 64, "nf4", torch.float32, out=output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@CUDA
def test_fp4_calls_original_backend(backend, monkeypatch):
    bnb, ops, original = backend
    torch.manual_seed(73)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    packed, state = bnb.functional.quantize_4bit(weight, quant_type="fp4")
    expected = bnb.functional.dequantize_4bit(packed, state)
    fallbacks = []

    def observed(*args, **kwargs):
        fallbacks.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(ops, "_dequantize_4bit_impl", observed)
    launches = _install_and_observe_launches(monkeypatch)
    actual = bnb.functional.dequantize_4bit(packed, state)
    assert fallbacks == ["fp4"] and launches == []
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
def test_noncontiguous_packed_storage_calls_original(backend, monkeypatch):
    _, ops, original = backend
    torch.manual_seed(74)
    packed = torch.randint(0, 256, (2, 128), device="cuda", dtype=torch.uint8).t()
    assert not packed.is_contiguous()
    scales = torch.rand(8, device="cuda")
    expected = torch.empty(512, device="cuda")
    original(packed, scales, 64, "nf4", torch.float32, out=expected)
    fallbacks = []

    def observed(*args, **kwargs):
        fallbacks.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(ops, "_dequantize_4bit_impl", observed)
    launches = _install_and_observe_launches(monkeypatch)
    output = torch.empty_like(expected)
    ops._dequantize_4bit_impl(packed, scales, 64, "nf4", torch.float32, out=output)
    assert fallbacks == [True] and launches == []
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@CUDA
def test_bnb_matmul_forward_and_backward_both_reach_custom_kernel(backend, monkeypatch):
    bnb, _, _ = backend
    torch.manual_seed(75)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    packed, state = bnb.functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
    # bnb 0.50.2 routes M > 1536 through its dequant+linear backend on CUDA.
    ref_x = torch.randn(2048, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    upstream = torch.randn(2048, 64, device="cuda", dtype=torch.float16)
    expected = bnb.matmul_4bit(ref_x, packed, quant_state=state)
    expected_dx = torch.autograd.grad(expected, ref_x, upstream)[0]
    launches = _install_and_observe_launches(monkeypatch)
    x = ref_x.detach().clone().requires_grad_(True)
    actual = bnb.matmul_4bit(x, packed, quant_state=state)
    actual_dx = torch.autograd.grad(actual, x, upstream)[0]
    assert launches == [weight.numel(), weight.numel()]
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=1e-3, atol=1e-3)


def test_patch_is_idempotent_and_cpu_falls_back_without_gpu_imports(monkeypatch):
    import solution.nf4 as implementation

    calls = []
    marker = object()

    def original(A, absmax, blocksize, quant_type, dtype, out):
        calls.append((A, absmax, blocksize, quant_type, dtype, out))
        out.fill_(7)
        return marker

    fake_backend = SimpleNamespace(_dequantize_4bit_impl=original)

    def load_backend(name):
        assert name == "bitsandbytes.backends.cuda.ops"
        return fake_backend

    monkeypatch.setattr(implementation, "import_module", load_backend)
    implementation.patch_nf4()
    first_patch = fake_backend._dequantize_4bit_impl
    implementation.patch_nf4()
    assert fake_backend._dequantize_4bit_impl is first_patch
    packed = torch.zeros(16, dtype=torch.uint8)
    scales = torch.ones(1)
    output = torch.empty(32)
    result = first_patch(packed, scales, 64, "nf4", torch.float32, out=output)
    assert result is marker and len(calls) == 1
    assert calls[0][0] is packed and calls[0][-1] is output
    torch.testing.assert_close(output, torch.full_like(output, 7))


def test_scale_cache_cpu_guard_and_bounded_weak_lifetime(monkeypatch):
    import gc
    import solution.nf4 as implementation

    implementation._clear_scale_cache()
    quantized = torch.arange(64, dtype=torch.uint8)
    absmax, code, offset = torch.ones(1), torch.arange(256).float(), torch.tensor(0.25)
    assert not implementation._scale_inputs_supported(quantized, absmax, code, offset)
    # Exercise ownership/invalidation logic on CPU independently of the CUDA
    # guard. The fake decoder is a numerical stand-in, never production code.
    monkeypatch.setattr(implementation, "_scale_inputs_supported", lambda *args: True)
    monkeypatch.setattr(implementation, "_SCALE_CACHE_LIMIT", 256)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=17))
    decodes = []

    def decode(q, scales, mapping, blocksize, dtype, out):
        decodes.append(q)
        out.copy_(mapping[q.long()] * scales.repeat_interleave(blocksize)[:q.numel()])

    fake_backend = SimpleNamespace(_dequantize_blockwise_impl=decode)
    try:
        first = implementation._cached_scales(fake_backend, quantized, absmax, code, offset)
        assert implementation._cached_scales(fake_backend, quantized, absmax, code, offset) is first
        assert len(decodes) == 1 and implementation._SCALE_CACHE_BYTES == 256
        offset.add_(1)
        updated = implementation._cached_scales(fake_backend, quantized, absmax, code, offset)
        torch.testing.assert_close(updated, first + 1, rtol=0, atol=0)
        assert len(decodes) == 2 and updated is not first
        another = quantized.clone()
        implementation._cached_scales(fake_backend, another, absmax, code, offset)
        assert len(implementation._SCALE_CACHE) == 1 and implementation._SCALE_CACHE_BYTES == 256
        decodes.clear()  # The observer must not keep the source tensor alive.
        del another
        gc.collect()
        assert not implementation._SCALE_CACHE and implementation._SCALE_CACHE_BYTES == 0
    finally:
        implementation._clear_scale_cache()


@pytest.fixture
def scale_cache_backend(backend, monkeypatch):
    import solution.nf4 as implementation

    bnb, ops, _ = backend
    for owner, name in ((ops, "_dequant_linear_fallback"), (bnb.functional, "dequantize_4bit")):
        function = getattr(owner, name)
        while getattr(function, "_soup_nf4_scale_cache", False):
            function = function.__wrapped__
        monkeypatch.setattr(owner, name, function)
    implementation._clear_scale_cache()
    yield bnb, ops
    implementation._clear_scale_cache()


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_shared_scale_cache_reuses_forward_backward_and_keeps_own_nf4_launches(
    scale_cache_backend, monkeypatch, dtype,
):
    import solution.nf4 as implementation

    bnb, ops = scale_cache_backend
    torch.manual_seed(177)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    packed, state = bnb.functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
    x = torch.randn(2048, 128, device="cuda", dtype=dtype, requires_grad=True)
    upstream = torch.randn(2048, 64, device="cuda", dtype=dtype)
    own_launches = _install_and_observe_launches(monkeypatch)
    expected = bnb.matmul_4bit(x, packed, quant_state=state)
    expected_gradient = torch.autograd.grad(expected, x, upstream)[0]
    own_launches.clear()
    decode_calls = []
    original_decode = ops._dequantize_blockwise_impl

    def observed_decode(*args, **kwargs):
        decode_calls.append(1)
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(ops, "_dequantize_blockwise_impl", observed_decode)
    assert implementation.patch_nf4_scale_cache()
    first_forward = ops._dequant_linear_fallback
    first_dequant = bnb.functional.dequantize_4bit
    assert implementation.patch_nf4_scale_cache()
    assert ops._dequant_linear_fallback is first_forward and bnb.functional.dequantize_4bit is first_dequant
    state_snapshot = (state.absmax, state.state2, state.offset, state.nested)
    rng = torch.cuda.get_rng_state().clone()
    for _ in range(2):
        actual_x = x.detach().clone().requires_grad_(True)
        actual = bnb.matmul_4bit(actual_x, packed, quant_state=state)
        gradient = torch.autograd.grad(actual, actual_x, upstream)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert len(decode_calls) == 1
    assert own_launches == [weight.numel()] * 4
    assert implementation._SCALE_CACHE_BYTES == state.absmax.numel() * 4
    assert len(implementation._SCALE_CACHE) == 1
    assert state.absmax is state_snapshot[0] and state.state2 is state_snapshot[1]
    assert state.offset is state_snapshot[2] and state.nested is state_snapshot[3]
    assert torch.equal(torch.cuda.get_rng_state(), rng)


@CUDA
@pytest.mark.parametrize("changed", ["quantized", "absmax", "code", "offset", "replacement"])
def test_scale_cache_invalidates_changed_source_tensors(scale_cache_backend, changed):
    import solution.nf4 as implementation

    bnb, ops = scale_cache_backend
    torch.manual_seed(178)
    packed, state = bnb.functional.quantize_4bit(
        torch.randn(64, 128, device="cuda", dtype=torch.float16),
        quant_type="nf4", compress_statistics=True,
    )

    def cached():
        return implementation._cached_scales(ops, state.absmax, state.state2.absmax,
                                             state.state2.code, state.offset)

    before = cached()
    assert cached() is before
    with torch.no_grad():
        if changed == "replacement":
            state.offset = state.offset.clone() + 0.125
        else:
            tensor = {"quantized": state.absmax, "absmax": state.state2.absmax,
                      "code": state.state2.code, "offset": state.offset}[changed]
            tensor.add_(1 if tensor.dtype == torch.uint8 else 0.125)
    after = cached()
    reference = bnb.functional.dequantize_blockwise(state.absmax, state.state2) + state.offset
    assert after is not before
    torch.testing.assert_close(after, reference, rtol=0, atol=0)
    assert cached() is after


@CUDA
def test_scale_cache_keeps_nonnested_and_fp4_paths_original(scale_cache_backend):
    import solution.nf4 as implementation

    bnb, _ = scale_cache_backend
    assert implementation.patch_nf4_scale_cache()
    for quant_type, nested in (("nf4", False), ("fp4", True)):
        weight = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        packed, state = bnb.functional.quantize_4bit(weight, quant_type=quant_type, compress_statistics=nested)
        expected = bnb.functional.dequantize_4bit.__wrapped__(packed, state)
        actual = bnb.functional.dequantize_4bit(packed, state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not implementation._SCALE_CACHE
