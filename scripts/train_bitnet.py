import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tiny_llm import BitNetLLM, SentencesDataset


def train_bitnet(
    corpus_path: str,
    tokenizer_path: str,
    output_path: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    print("⚡ Starting 1.58-Bit Quantization-Aware Training (BitNet b1.58)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:     {device}")
    print(f"  Corpus:     {corpus_path}")
    print(f"  Tokenizer:  {tokenizer_path}")

    if not os.path.exists(corpus_path) or not os.path.exists(tokenizer_path):
        print(
            "❌ Dataset or Tokenizer not found. Please run scripts/prepare_data.py first."
        )
        return

    # Load dataset
    dataset = SentencesDataset(
        file_path=corpus_path, tokenizer_path=tokenizer_path, max_length=64
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = dataset.tokenizer.get_vocab_size()

    # Instantiate 1.58-bit BitNetLLM
    model = BitNetLLM(
        vocab_size=vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        ffn_dim=512,
        max_seq_len=64,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  BitNet Parameters: {total_params:,} (Ternary Weights in {{-1, 0, +1}})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.token_to_id("[PAD]"))

    model.train()
    print("\nStarting QAT Training Loop...")
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
        print(f"  Epoch [{epoch}/{epochs}] - BitNet QAT Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"✅ 1.58-bit BitNet training complete! Saved checkpoint to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1.58-bit BitNet b1.58 Model.")
    parser.add_argument(
        "--corpus", type=str, default="data/corpus.txt", help="Path to text corpus"
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
        default="checkpoints/bitnet_llm.pth",
        help="Output model checkpoint path",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    train_bitnet(
        corpus_path=args.corpus,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
