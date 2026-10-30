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
