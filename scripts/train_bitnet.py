import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tiny_llm import TinyLLM, BitNetLLM, SentencesDataset


def train_bitnet(
    corpus_path: str,
    tokenizer_path: str,
    output_path: str,
    teacher_path: str = None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-3,
    alpha: float = 0.5,
):
    print("⚡ Starting 1.58-Bit Quantization-Aware Training (BitNet b1.58)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:         {device}")
    print(f"  Corpus:         {corpus_path}")
    print(f"  Tokenizer:      {tokenizer_path}")
    print(
        f"  Teacher Model:  {teacher_path if teacher_path else 'None (Training from Scratch)'}"
    )
    print(f"  Learning Rate:  {lr} (Higher LR for STE threshold crossing)")
    print("  Weight Decay:   0.0 (Prevents ternary trapping at 0)")

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

    # Instantiate student BitNetLLM model
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

    # Optional Teacher Model loading
    teacher_model = None
    if teacher_path and os.path.exists(teacher_path):
        print(f"  🎓 Loading Float32 Teacher Model from '{teacher_path}'...")
        teacher_model = TinyLLM(
            vocab_size=vocab_size,
            dim=128,
            n_layers=4,
            n_heads=4,
            ffn_dim=512,
            max_seq_len=64,
        ).to(device)
        teacher_model.load_state_dict(
            torch.load(teacher_path, map_location=device, weights_only=True)
        )
        teacher_model.eval()

    # Disable weight decay to prevent trapping master weights at zero
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion_ce = nn.CrossEntropyLoss(
        ignore_index=dataset.tokenizer.token_to_id("[PAD]")
    )

    model.train()
    total_steps = epochs * len(dataloader)
    current_step = 0

    print("\nStarting QAT Training Loop...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for x, y in dataloader:
            current_step += 1
            x, y = x.to(device), y.to(device)

            # Linear Learning Rate Warmup for first 10% steps
            warmup_steps = max(10, int(total_steps * 0.1))
            if current_step <= warmup_steps:
                warmup_lr = lr * (current_step / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            optimizer.zero_grad()
            student_logits = model(x)

            # 1. Hard CrossEntropy Loss
            loss = criterion_ce(student_logits.view(-1, vocab_size), y.view(-1))

            # 2. Optional Teacher Soft Logits Distillation Loss
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(x)
                distill_loss = F.mse_loss(student_logits, teacher_logits)
                loss = (1.0 - alpha) * loss + alpha * distill_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch [{epoch}/{epochs}] - BitNet QAT Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(
        f"\n✅ 1.58-bit BitNet training complete! Saved checkpoint to '{output_path}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 1.58-bit BitNet b1.58 Model with Distillation."
    )
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
        "--teacher",
        type=str,
        default="checkpoints/tiny_llm.pth",
        help="Optional Float32 teacher model checkpoint for distillation",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Learning rate (default: 3e-3 for STE crossing)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Teacher distillation loss weight factor",
    )

    args = parser.parse_args()

    train_bitnet(
        corpus_path=args.corpus,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        teacher_path=args.teacher,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
    )
