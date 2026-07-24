import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tiny_llm import TinyLLM, SentencesDataset, inject_lora, merge_lora


def train_lora():
    print("🚀 Initializing LoRA Fine-Tuning...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    model_path = (
        "checkpoints/tiny_llm.pth"
        if os.path.exists("checkpoints/tiny_llm.pth")
        else "tiny_llm.pth"
    )
    tokenizer_path = (
        "checkpoints/tokenizer.json"
        if os.path.exists("checkpoints/tokenizer.json")
        else "tokenizer.json"
    )
    corpus_path = (
        "data/corpus.txt" if os.path.exists("data/corpus.txt") else "corpus.txt"
    )

    if not os.path.exists(corpus_path) or not os.path.exists(tokenizer_path):
        print(
            "❌ Dataset or Tokenizer not found. Please run scripts/prepare_data.py first."
        )
        return

    # Load dataset
    dataset = SentencesDataset(
        file_path=corpus_path, tokenizer_path=tokenizer_path, max_length=64
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    vocab_size = dataset.tokenizer.get_vocab_size()

    # Load base model
    model = TinyLLM(
        vocab_size=vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        ffn_dim=512,
        max_seq_len=64,
    )
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        print(f"Loaded base model checkpoint from {model_path}")

    # Inject LoRA adapters (rank r=4 into Query 'wq' and Value 'wv' projections)
    model = inject_lora(model, r=4, lora_alpha=1.0, target_modules=("wq", "wv"))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total_params:,}")
    print(
        f"Trainable Parameters: {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}% of total)"
    )

    # Optimizer: Only update trainable LoRA parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.token_to_id("[PAD]"))

    # Fine-tuning loop (5 epochs for demonstration)
    model.train()
    print("\nStarting LoRA Fine-Tuning Loop...")
    for epoch in range(1, 6):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/5 - LoRA Loss: {avg_loss:.4f}")

    # Save small LoRA adapters dictionary
    os.makedirs("checkpoints", exist_ok=True)
    adapter_state = {
        k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k
    }
    torch.save(adapter_state, "checkpoints/lora_adapters.pth")
    print(
        f"✅ Saved LoRA adapter weights ({len(adapter_state)} tensors) to checkpoints/lora_adapters.pth"
    )

    # Demonstrate merging LoRA weights back into standard linear layers for zero-latency inference
    merged_model = merge_lora(model)
    torch.save(merged_model.state_dict(), "checkpoints/tiny_llm_finetuned.pth")
    print(
        "✅ Merged LoRA weights into base model and saved to checkpoints/tiny_llm_finetuned.pth"
    )


if __name__ == "__main__":
    train_lora()
