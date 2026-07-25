import time
import torch
import torch.nn as nn
from tiny_llm import TinyLLM, inject_lora


def run_lora_benchmark():
    print("=" * 80)
    print("📊 BENCHMARK 4: Full Fine-Tuning vs. LoRA (Low-Rank Adaptation, r=4)")
    print("=" * 80)

    vocab_size = 1000
    dim = 128
    n_layers = 4
    batch_size = 16
    seq_len = 32

    # 1. Full Fine-Tuning Model Setup
    full_model = TinyLLM(vocab_size=vocab_size, dim=dim, n_layers=n_layers)
    full_total_params = sum(p.numel() for p in full_model.parameters())
    full_trainable_params = sum(
        p.numel() for p in full_model.parameters() if p.requires_grad
    )

    # 2. LoRA Model Setup
    lora_model = TinyLLM(vocab_size=vocab_size, dim=dim, n_layers=n_layers)
    lora_model = inject_lora(lora_model, r=4, target_modules=("wq", "wv"))
    lora_total_params = sum(p.numel() for p in lora_model.parameters())
    lora_trainable_params = sum(
        p.numel() for p in lora_model.parameters() if p.requires_grad
    )

    dummy_x = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_y = torch.randint(0, vocab_size, (batch_size, seq_len))
    criterion = nn.CrossEntropyLoss()

    # Benchmark Full Fine-Tuning Step
    full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=1e-3)
    full_model.train()
    t0 = time.perf_counter()
    for _ in range(10):
        full_optimizer.zero_grad()
        out = full_model(dummy_x)
        loss = criterion(out.view(-1, vocab_size), dummy_y.view(-1))
        loss.backward()
        full_optimizer.step()
    full_time = (time.perf_counter() - t0) / 10.0

    # Benchmark LoRA Step
    lora_optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad], lr=1e-3
    )
    lora_model.train()
    t0 = time.perf_counter()
    for _ in range(10):
        lora_optimizer.zero_grad()
        out = lora_model(dummy_x)
        loss = criterion(out.view(-1, vocab_size), dummy_y.view(-1))
        loss.backward()
        lora_optimizer.step()
    lora_time = (time.perf_counter() - t0) / 10.0

    print(
        f"  Training Strategy  | {'Total Params':<14} | {'Trainable Params':<18} | {'% Trainable':<12} | {'Step Time (ms)':<14}"
    )
    print("-" * 80)
    print(
        f"  Full Fine-Tuning   | {full_total_params:<14,} | {full_trainable_params:<18,} | {(full_trainable_params / full_total_params) * 100:<11.2f}% | {full_time * 1000:<14.2f}"
    )
    print(
        f"  LoRA Adapter (r=4) | {lora_total_params:<14,} | {lora_trainable_params:<18,} | {(lora_trainable_params / lora_total_params) * 100:<11.2f}% | {lora_time * 1000:<14.2f}"
    )
    print("-" * 80)
    print(
        f"💡 Key Takeaway: LoRA updates only {lora_trainable_params:,} parameters ({(lora_trainable_params / lora_total_params) * 100:.2f}% of model),"
    )
    print(
        "   saving substantial memory during backward pass and optimizer state tracking."
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_lora_benchmark()
