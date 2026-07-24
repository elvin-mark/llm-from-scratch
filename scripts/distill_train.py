import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tiny_llm import TinyLLM, MoELLM, MoELLMConfig, SentencesDataset


def train_distilled_student(
    corpus_path: str,
    tokenizer_path: str,
    output_path: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    use_moe: bool = False,
):
    print("🎓 Starting Knowledge Distillation Student Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:         {device}")
    print(f"  Distilled Data: {corpus_path}")
    print(f"  Model Type:     {'MoELLM (GQA + MoE)' if use_moe else 'TinyLLM (Dense)'}")

    if not os.path.exists(corpus_path):
        print(f"❌ Distilled corpus file '{corpus_path}' not found.")
        print(
            "💡 Run `python scripts/distill_generate.py` first to generate synthetic text."
        )
        return

    if not os.path.exists(tokenizer_path):
        print(
            f"❌ Tokenizer file '{tokenizer_path}' not found. Please train tokenizer first."
        )
        return

    # Load dataset
    dataset = SentencesDataset(
        file_path=corpus_path, tokenizer_path=tokenizer_path, max_length=64
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = dataset.tokenizer.get_vocab_size()

    # Instantiate student model architecture
    if use_moe:
        config = MoELLMConfig(
            vocab_size=vocab_size,
            dim=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            ffn_dim=512,
            num_experts=4,
            num_experts_per_tok=2,
            max_seq_len=64,
        )
        model = MoELLM(config=config).to(device)
    else:
        model = TinyLLM(
            vocab_size=vocab_size,
            dim=128,
            n_layers=4,
            n_heads=4,
            ffn_dim=512,
            max_seq_len=64,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Student Model Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.token_to_id("[PAD]"))

    model.train()
    for epoch in range(1, epochs + 1):
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
        print(f"  Epoch [{epoch}/{epochs}] - Distillation Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"✅ Student model training complete! Saved to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Student Model on Distilled Corpus."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/distilled_corpus.txt",
        help="Path to distilled synthetic corpus",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="checkpoints/tokenizer.json",
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/tiny_llm_distilled.pth",
        help="Output model checkpoint path",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--use-moe",
        action="store_true",
        help="Train a MoELLM student instead of TinyLLM",
    )

    args = parser.parse_args()

    train_distilled_student(
        corpus_path=args.corpus,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_moe=args.use_moe,
    )
