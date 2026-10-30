"""Verify the MLP storage change against v2's unchanged SwiGLU path.

CUDA tests use small real bitsandbytes NF4 layers, including double quantization.
The reference preserves the original fp32 gate/up storage and its cast nodes.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
HIDDEN = 128
INTERMEDIATE = 256


def _make_mlp():
    bnb = pytest.importorskip("bitsandbytes")
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from transformers import Qwen2Config

    config = Qwen2Config(
        hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_attention_heads=4, num_key_value_heads=2, hidden_act="silu",
    )
    module = modeling.Qwen2MLP(config)

    def projection(inputs, outputs):
        layer = bnb.nn.Linear4bit(
            inputs, outputs, bias=False, compute_dtype=torch.float16,
            quant_type="nf4", compress_statistics=True,
        )
        with torch.no_grad():
            layer.weight.copy_(torch.randn_like(layer.weight) / inputs ** 0.5)
        # Match the task's original half weights, then NF4 quantization on CUDA.
        layer = layer.to(dtype=torch.float16).cuda()
        layer.weight.requires_grad_(False)
        assert layer.compute_dtype == torch.float16
        assert layer.compute_type_is_set is True
        return layer

    module.gate_proj = projection(HIDDEN, INTERMEDIATE)
    module.up_proj = projection(HIDDEN, INTERMEDIATE)
    module.down_proj = projection(INTERMEDIATE, HIDDEN)
    return module, modeling


def _v2_forward(module, x):
    from solution.swiglu import swiglu

    return module.down_proj(swiglu(module.gate_proj(x), module.up_proj(x)))


@CUDA
@pytest.mark.parametrize("shape", [(17, HIDDEN), (2, 19, HIDDEN)])
@pytest.mark.parametrize("autocast", [False, True])
def test_half_storage_preserves_output_and_fp32_input_gradient(shape, autocast):
    from solution.mlp import mlp_forward, supported_mlp

    torch.manual_seed(91)
    module, _ = _make_mlp()
    reference_x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    x = reference_x.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(x)
    assert supported_mlp(module, x)
    saved_reference, saved_optimized = [], []

    def capture(destination):
        def save(tensor):
            destination.append((tuple(tensor.shape), tensor.dtype))
            return tensor
        return save

    def amp_context():
        return torch.autocast("cuda", dtype=torch.float16) if autocast else nullcontext()

    with amp_context(), torch.autograd.graph.saved_tensors_hooks(capture(saved_reference), lambda t: t):
        expected = _v2_forward(module, reference_x)
    reference_dx = torch.autograd.grad(expected, reference_x, upstream)[0]
    with amp_context(), torch.autograd.graph.saved_tensors_hooks(capture(saved_optimized), lambda t: t):
        actual = mlp_forward(module, x)
    dx = torch.autograd.grad(actual, x, upstream)[0]

    assert actual.dtype == torch.float32 and dx.dtype == torch.float32
    # Loose fp16 tolerances could hide an incorrect shared input cast: the two
    # branch gradients must be widened independently before their fp32 sum.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(dx, reference_dx, rtol=0, atol=0)
    activation_shape = (*shape[:-1], INTERMEDIATE)
    assert saved_reference.count((activation_shape, torch.float32)) >= 2
    assert saved_optimized.count((activation_shape, torch.float16)) >= 2
    assert (activation_shape, torch.float32) not in saved_optimized


@CUDA
def test_actual_qwen_patch_matches_v2_and_is_idempotent(monkeypatch):
    import solution.mlp as implementation

    torch.manual_seed(92)
    module, modeling = _make_mlp()
    # Give the patch precisely the v2 reference path as its fallback, restoring
    # the incoming global method at test exit regardless of earlier patches.
    monkeypatch.setattr(modeling.Qwen2MLP, "forward", _v2_forward)
    reference_x = torch.randn(2, 17, HIDDEN, device="cuda", requires_grad=True)
    expected = module(reference_x)
    upstream = torch.randn_like(expected)
    reference_dx = torch.autograd.grad(expected, reference_x, upstream)[0]
    implementation.patch_qwen2_mlp_dtype()
    first_patch = modeling.Qwen2MLP.forward
    implementation.patch_qwen2_mlp_dtype()
    assert modeling.Qwen2MLP.forward is first_patch

    observed_inputs = {}

    def capture(name):
        def observe(layer, args):
            observed_inputs[name] = args[0]
        return observe

    hooks = [module.gate_proj.register_forward_pre_hook(capture("gate")),
             module.up_proj.register_forward_pre_hook(capture("up"))]
    try:
        x = reference_x.detach().clone().requires_grad_(True)
        actual = module(x)
        dx = torch.autograd.grad(actual, x, upstream)[0]
    finally:
        for hook in hooks:
            hook.remove()
    assert observed_inputs["gate"].dtype == observed_inputs["up"].dtype == torch.float16
    assert observed_inputs["gate"] is not observed_inputs["up"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(dx, reference_dx, rtol=0, atol=0)


class _LoRAWrapper(torch.nn.Module):
    """Small valid adapter wrapper: the patch must preserve this extra branch."""

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = torch.nn.Linear(HIDDEN, 2, bias=False, device="cuda")
        self.lora_B = torch.nn.Linear(2, INTERMEDIATE, bias=False, device="cuda")

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x))


@CUDA
@pytest.mark.parametrize("unsupported", [
    "unset_compute_type", "fp32_compute", "bias", "other_activation",
    "trainable_projection", "lora_wrapper",
])
def test_unsupported_mlp_calls_the_original_forward(monkeypatch, unsupported):
    from solution.mlp import patch_qwen2_mlp_dtype, supported_mlp

    torch.manual_seed(93)
    module, modeling = _make_mlp()
    if unsupported == "unset_compute_type":
        module.gate_proj.compute_type_is_set = False
    elif unsupported == "fp32_compute":
        module.gate_proj.compute_dtype = torch.float32
    elif unsupported == "bias":
        module.gate_proj.bias = torch.nn.Parameter(torch.zeros(INTERMEDIATE, device="cuda"), requires_grad=False)
    elif unsupported == "other_activation":
        module.config.hidden_act = "gelu"
        module.act_fn = torch.nn.GELU()
    elif unsupported == "trainable_projection":
        module.gate_proj = torch.nn.Linear(HIDDEN, INTERMEDIATE, bias=False, device="cuda")
    elif unsupported == "lora_wrapper":
        module.gate_proj = _LoRAWrapper(module.gate_proj)

    x = torch.randn(2, 17, HIDDEN, device="cuda", requires_grad=True)
    assert not supported_mlp(module, x)
    calls = []

    def original(self, value):
        calls.append((self, value))
        return value + 3

    monkeypatch.setattr(modeling.Qwen2MLP, "forward", original)
    patch_qwen2_mlp_dtype()
    output = module(x)
    assert len(calls) == 1 and calls[0][0] is module and calls[0][1] is x
    torch.testing.assert_close(output, x + 3, rtol=0, atol=0)


def test_cpu_guard_needs_no_gpu_packages():
    from solution.mlp import supported_mlp

    # Reject CPU tensors before importing optional CUDA/model frameworks.
    assert not supported_mlp(SimpleNamespace(), torch.randn(2, 3, HIDDEN))
