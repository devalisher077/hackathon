"""Manual frozen-MLP autograd versus the existing real bitsandbytes path."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _finite_close(actual, expected, *, label):
    assert torch.isfinite(expected).all(), f"{label}: reference is nonfinite"
    assert torch.isfinite(actual).all(), f"{label}: optimized is nonfinite"
    difference = actual.float() - expected.float()
    reference = expected.float()
    relative = difference.norm() / reference.norm().clamp_min(1e-30)
    assert relative < 5e-3, (label, float(relative))
    maximum = difference.abs().max()
    assert maximum < 5e-3 * reference.abs().max() + 1e-7, (label, float(maximum))


@CUDA
@pytest.mark.parametrize("shape", [(65, 128), (2, 49, 128)])
@pytest.mark.parametrize("autocast", [False, True])
def test_real_nf4_forward_scaled_dx_and_saved_storage(shape, autocast):
    from solution.test_mlp import _make_mlp
    from solution.mlp import mlp_forward
    from solution.fused_mlp import supported_fused_mlp, fused_mlp_forward

    torch.manual_seed(261)
    module, _ = _make_mlp()
    x = torch.randn(shape, device="cuda", requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    upstream = torch.randn_like(x) * (128 / 128 ** 0.5)
    assert supported_fused_mlp(module, x)
    assert all(not parameter.requires_grad for parameter in module.parameters())
    saved = []

    def save(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size()))
        return tensor

    def amp():
        return torch.autocast("cuda", dtype=torch.float16) if autocast else nullcontext()

    with amp():
        expected = mlp_forward(module, reference_x)
    expected_dx = torch.autograd.grad(expected, reference_x, upstream)[0]
    rng = torch.cuda.get_rng_state().clone()
    with amp(), torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        actual = fused_mlp_forward(module, x)
    actual_dx = torch.autograd.grad(actual, x, upstream)[0]
    assert actual.shape == actual_dx.shape == x.shape
    assert actual.dtype == actual_dx.dtype == torch.float32
    _finite_close(actual, expected, label="MLP output")
    _finite_close(actual_dx, expected_dx, label="scaled input gradient")
    # A shared differentiable half cast or a half accumulation of the branch
    # gradients would force every resulting fp32 value onto the half grid.
    assert not torch.equal(actual_dx, actual_dx.half().float())
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert all(parameter.grad is None for parameter in module.parameters())
    rows = x.numel() // x.shape[-1]
    assert len(saved) == 2
    assert all(dtype == torch.float16 and shape[-1] == 256 for shape, dtype, _ in saved)
    assert sum(size for _, _, size in saved) == 2 * rows * 256 * 2


@CUDA
def test_retain_graph_keeps_saved_tensors_until_the_final_backward():
    from solution.test_mlp import _make_mlp
    from solution.mlp import mlp_forward
    from solution.fused_mlp import fused_mlp_forward

    torch.manual_seed(262)
    module, _ = _make_mlp()
    x = torch.randn(67, 128, device="cuda", requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        actual, expected = fused_mlp_forward(module, x), mlp_forward(module, reference_x)
    for retain in (True, True, False):
        upstream = torch.randn_like(x) * 8
        actual_dx = torch.autograd.grad(actual, x, upstream, retain_graph=retain)[0]
        reference_dx = torch.autograd.grad(expected, reference_x, upstream, retain_graph=retain)[0]
        _finite_close(actual_dx, reference_dx, label=f"retain_graph={retain}")
    with pytest.raises(RuntimeError, match="backward through the graph a second time|saved tensors"):
        torch.autograd.grad(actual, x, torch.ones_like(actual))


def test_cpu_guard_rejects_before_importing_gpu_dependencies():
    from solution.fused_mlp import supported_fused_mlp

    assert not supported_fused_mlp(SimpleNamespace(), torch.randn(2, 3, 128, requires_grad=True))


@CUDA
@pytest.mark.parametrize("case", [
    "eval", "no_input_gradient", "half_input", "noncontiguous", "fp32_compute", "bias",
    "trainable_projection", "lora_wrapper", "other_activation", "projection_hook", "tiny_m",
])
def test_guarded_cases_call_original_without_bypassing_module_behavior(monkeypatch, case):
    from solution.test_mlp import _make_mlp, _LoRAWrapper
    from solution.fused_mlp import supported_fused_mlp, patch_qwen2_fused_mlp

    torch.manual_seed(263)
    module, modeling = _make_mlp()
    x = torch.randn(2, 33, 128, device="cuda", requires_grad=True)
    hook = None
    if case == "eval":
        module.eval()
    elif case == "no_input_gradient":
        x = x.detach()
    elif case == "half_input":
        x = x.detach().half().requires_grad_()
    elif case == "noncontiguous":
        x = torch.randn(2, 33, 256, device="cuda", requires_grad=True)[..., ::2]
    elif case == "fp32_compute":
        module.up_proj.compute_dtype = torch.float32
    elif case == "bias":
        module.gate_proj.bias = torch.nn.Parameter(torch.zeros(256, device="cuda"), requires_grad=False)
    elif case == "trainable_projection":
        module.gate_proj = torch.nn.Linear(128, 256, bias=False, device="cuda")
    elif case == "lora_wrapper":
        module.gate_proj = _LoRAWrapper(module.gate_proj)
    elif case == "other_activation":
        module.config.hidden_act = "gelu"
        module.act_fn = torch.nn.GELU()
    elif case == "projection_hook":
        hook = module.gate_proj.register_forward_pre_hook(lambda module, args: None)
    elif case == "tiny_m":
        x = torch.randn(17, 128, device="cuda", requires_grad=True)
    try:
        assert not supported_fused_mlp(module, x)
        calls = []

        def original(self, value):
            calls.append((self, value))
            return value + 3

        monkeypatch.setattr(modeling.Qwen2MLP, "forward", original)
        patch_qwen2_fused_mlp()
        first = modeling.Qwen2MLP.forward
        patch_qwen2_fused_mlp()
        assert modeling.Qwen2MLP.forward is first
        output = module(x)
        assert len(calls) == 1 and calls[0][0] is module and calls[0][1] is x
        torch.testing.assert_close(output, x + 3, rtol=0, atol=0)
    finally:
        if hook is not None:
            hook.remove()


@CUDA
def test_actual_nf4_qwen_lora_with_padding_preserves_loss_gradients_and_rng(monkeypatch):
    from solution import fused_mlp
    from solution.mlp import mlp_forward
    from solution.padded_mlp import patch_padding_mlp
    from solution.test_padded_mlp import (
        _unpatched_model_classes, _make_quantized_lora_model, _batch,
        _rng_state, _restore_rng, _metrics,
    )

    torch.manual_seed(264)
    modeling = _unpatched_model_classes(monkeypatch, v3=True)
    original_model = modeling.Qwen2Model.forward
    # Explicitly choose the current non-manual reference even if apply()
    # previously installed this candidate on the global Qwen MLP class.
    monkeypatch.setattr(modeling.Qwen2MLP, "forward", mlp_forward)
    model, core, trainable = _make_quantized_lora_model(modeling)
    tokens, mask, labels = _batch((41, 37, 19), device="cuda")
    patch_padding_mlp()
    initial_rng = _rng_state()
    with torch.autocast("cuda", dtype=torch.float16):
        expected = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
    expected_gradient = torch.autograd.grad(expected.loss * 128, [parameter for _, parameter in trainable])
    expected_rng = _rng_state()
    expected_metrics = _metrics(expected.logits, mask, labels)

    # Install in production order: manual MLP first, then outer compaction.
    monkeypatch.setattr(modeling.Qwen2Model, "forward", original_model)
    monkeypatch.setattr(modeling.Qwen2MLP, "forward", mlp_forward)
    fused_mlp.patch_qwen2_fused_mlp()
    first = modeling.Qwen2MLP.forward
    fused_mlp.patch_qwen2_fused_mlp()
    assert modeling.Qwen2MLP.forward is first
    patch_padding_mlp()
    real_forward = fused_mlp.fused_mlp_forward
    calls = []

    def observed_forward(module, x):
        calls.append(tuple(x.shape))
        return real_forward(module, x)

    monkeypatch.setattr(fused_mlp, "fused_mlp_forward", observed_forward)
    _restore_rng(initial_rng)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False)
    actual_gradient = torch.autograd.grad(actual.loss * 128, [parameter for _, parameter in trainable])
    actual_rng = _rng_state()
    assert calls == [(97, 128)] * len(core.layers), "Compaction and the manual MLP must both execute"
    assert actual.keys() == expected.keys()
    _finite_close(actual.logits, expected.logits, label="full Qwen logits")
    _finite_close(actual.loss, expected.loss, label="full Qwen loss")
    for (name, _), gradient, ref_gradient in zip(trainable, actual_gradient, expected_gradient):
        assert ref_gradient.norm() > 0, name
        _finite_close(gradient, ref_gradient, label=name)
    for actual_metric, ref_metric in zip(_metrics(actual.logits, mask, labels), expected_metrics):
        torch.testing.assert_close(actual_metric, ref_metric, rtol=3e-3, atol=2e-5)
    assert torch.equal(actual_rng[0], expected_rng[0])
    assert torch.equal(actual_rng[1], expected_rng[1])
