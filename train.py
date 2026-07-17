import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from data import SentencesDataset
from model import TinyLLM


def train(tokenizer_path: str = "tokenizer.json", corpus_path="corpus.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SentencesDataset(corpus_path, tokenizer_path, max_length=64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    model = TinyLLM(
        vocab_size=vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        ffn_dim=512,
        max_seq_len=64,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pad_idx = tokenizer.token_to_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    epochs = 10
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "tiny_llm.pth")
    print("Model saved to tiny_llm.pth")


if __name__ == "__main__":
    train()
