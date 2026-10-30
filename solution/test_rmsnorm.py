"""RMSNorm numerical and integration checks; CUDA cases run on Kaggle T4.

Run ``python -m pytest solution/test_rmsnorm.py -v``. CPU fallback tests do
not validate the Triton implementation; all CUDA cases must also pass.
"""

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _reference(x, weight, eps):
    # The operation order and intermediate cast are part of Qwen2's semantics.
    value = x.float()
    variance = value.square().mean(dim=-1, keepdim=True)
    normalized = value * torch.rsqrt(variance + eps)
    return weight * normalized.to(x.dtype)


def _check_forward_backward(x, weight, eps=1e-6, operation=None):
    from solution.rmsnorm import rmsnorm

    if operation is None:
        operation = rmsnorm
    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(weight.requires_grad)
    expected = _reference(ref_x, ref_weight, eps)
    actual = operation(x, weight, eps)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)

    upstream = torch.randn_like(expected)
    actual_inputs = (x, weight) if weight.requires_grad else (x,)
    expected_inputs = (ref_x, ref_weight) if weight.requires_grad else (ref_x,)
    gradients = torch.autograd.grad(actual, actual_inputs, grad_outputs=upstream)
    ref_gradients = torch.autograd.grad(expected, expected_inputs, grad_outputs=upstream)
    for gradient, ref_gradient in zip(gradients, ref_gradients):
        torch.testing.assert_close(gradient, ref_gradient, rtol=3e-4, atol=2e-4)


@CUDA
@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 5, 1536), (1536,), (3, 17), (3, 1001)])
def test_cuda_fp32_forward_and_input_gradient(shape):
    torch.manual_seed(31)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    # Non-unit weights detect an omitted or misplaced scale in both directions.
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    _check_forward_backward(x, weight)


@CUDA
@pytest.mark.parametrize("scale", [0.0, 1e-7])
def test_cuda_epsilon_dominates_near_zero(scale):
    torch.manual_seed(32)
    x = (torch.randn(3, 1536, device="cuda") * scale).requires_grad_(True)
    weight = torch.randn(1536, device="cuda")
    _check_forward_backward(x, weight, eps=1e-6)


@CUDA
def test_cuda_noncontiguous_input_and_weight():
    torch.manual_seed(33)
    x = torch.randn(1536, 7, device="cuda").t().detach().requires_grad_(True)
    weight = torch.randn(3072, device="cuda")[::2]
    assert not x.is_contiguous() and not weight.is_contiguous()
    _check_forward_backward(x, weight)


@CUDA
def test_cuda_autocast_keeps_fp32_norm_semantics():
    torch.manual_seed(34)
    x = torch.randn(2, 17, 1536, device="cuda", requires_grad=True)
    weight = torch.randn(1536, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _check_forward_backward(x, weight)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("trainable_weight", [False, True])
def test_cpu_fallback_preserves_dtype_and_all_gradients(dtype, trainable_weight):
    torch.manual_seed(35)
    x = torch.randn(3, 17, dtype=dtype, requires_grad=True)
    weight = torch.randn(17, dtype=torch.float32, requires_grad=trainable_weight)
    _check_forward_backward(x, weight)


@CUDA
def test_cuda_fp16_input_fp32_weight_fallback():
    torch.manual_seed(36)
    x = torch.randn(3, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(1536, device="cuda", dtype=torch.float32)
    _check_forward_backward(x, weight)


@CUDA
def test_cuda_trainable_weight_fallback_preserves_weight_gradient():
    torch.manual_seed(37)
    x = torch.randn(3, 1536, device="cuda", requires_grad=True)
    weight = torch.randn(1536, device="cuda", requires_grad=True)
    _check_forward_backward(x, weight)


@CUDA
def test_patch_reaches_triton_and_matches_original_qwen2(monkeypatch):
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    import solution.rmsnorm as implementation

    norm_class = modeling.Qwen2RMSNorm
    original_forward = norm_class.forward
    if getattr(original_forward, "_soup_custom_rmsnorm", False):
        original_forward = original_forward.__wrapped__
    # Restore the incoming class state after the test, even if another caller
    # already applied the solution; the numerical reference must stay unpatched.
    monkeypatch.setattr(norm_class, "forward", original_forward)
    norm = norm_class(1536, eps=1e-6).cuda().float()
    norm.weight.requires_grad_(False)
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
    x = torch.randn(3, 1536, device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    expected = original_forward(norm, ref_x)

    implementation.patch_qwen2_rmsnorm()
    first_patch = norm_class.forward
    implementation.patch_qwen2_rmsnorm()
    assert norm_class.forward is first_patch, "Repeated apply() must not wrap again"

    actual = norm(x)
    # A custom autograd node proves that this call did not silently fall back
    # to the original chain of PyTorch operators.
    assert actual.grad_fn is not None
    assert "RMSNorm" in type(actual.grad_fn).__name__
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
    upstream = torch.randn_like(actual)
    dx = torch.autograd.grad(actual, x, upstream)[0]
    ref_dx = torch.autograd.grad(expected, ref_x, upstream)[0]
    torch.testing.assert_close(dx, ref_dx, rtol=3e-4, atol=2e-4)


@pytest.mark.parametrize("trainable_weight", [False, True])
def test_patch_cpu_fallback_calls_original(monkeypatch, trainable_weight):
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution.rmsnorm import patch_qwen2_rmsnorm

    norm_class = modeling.Qwen2RMSNorm
    original_forward = norm_class.forward
    if getattr(original_forward, "_soup_custom_rmsnorm", False):
        original_forward = original_forward.__wrapped__
    calls = []

    def observed_original(self, x):
        calls.append(x.device.type)
        return original_forward(self, x)

    monkeypatch.setattr(norm_class, "forward", observed_original)
    patch_qwen2_rmsnorm()
    norm = norm_class(17, eps=1e-6).float()
    norm.weight.requires_grad_(trainable_weight)
    x = torch.randn(3, 17, requires_grad=True)
    _check_forward_backward(x, norm.weight, operation=lambda value, weight, eps: norm(value))
    assert calls == ["cpu"]
