import torch
import torch.nn as nn
from tiny_llm import TinyLLM, LoRALinear, inject_lora, merge_lora


def test_lora_step_zero_equivalence():
    """Verify that LoRALinear at step 0 (B=0) outputs identical results to base linear layer."""
    base_linear = nn.Linear(32, 64)
    lora_linear = LoRALinear(base_linear, r=4, lora_alpha=1.0)

    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        out_base = base_linear(x)
        out_lora = lora_linear(x)

    assert torch.allclose(out_base, out_lora, atol=1e-6), (
        "Step 0 LoRALinear output must equal base linear layer."
    )


def test_inject_lora_parameter_freezing():
    """Verify that inject_lora freezes base parameters and only enables gradients for LoRA A & B."""
    model = TinyLLM(vocab_size=100, dim=32, n_layers=2, n_heads=2, ffn_dim=64)

    # Inject LoRA into wq and wv
    model = inject_lora(model, r=4, target_modules=("wq", "wv"))

    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_params = [
        name for name, p in model.named_parameters() if not p.requires_grad
    ]

    assert len(trainable_params) > 0, "There should be trainable LoRA parameters."
    assert all("lora_A" in p or "lora_B" in p for p in trainable_params), (
        "Only lora_A and lora_B should be trainable."
    )
    assert any("tok_embeddings" in p for p in frozen_params), (
        "Base embeddings should be frozen."
    )
    assert any("output" in p for p in frozen_params), (
        "Base output projection should be frozen."
    )


def test_merge_lora_equivalence():
    """Verify that merging LoRA adapters yields mathematically identical outputs to the adapter model."""
    model = TinyLLM(vocab_size=100, dim=32, n_layers=2, n_heads=2, ffn_dim=64)
    model = inject_lora(model, r=4, target_modules=("wq", "wv"))

    # Simulate adapter training by writing non-zero values into lora_B
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.normal_(module.lora_B, std=0.02)

    model.eval()
    dummy_input = torch.randint(0, 100, (2, 8))

    with torch.no_grad():
        logits_adapter = model(dummy_input)

    # Merge LoRA weights into base weights
    model_merged = merge_lora(model)
    model_merged.eval()

    with torch.no_grad():
        logits_merged = model_merged(dummy_input)

    assert torch.allclose(logits_adapter, logits_merged, atol=1e-5), (
        "Merged model logits must match LoRA adapter model."
    )
