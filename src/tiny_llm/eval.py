import math

import torch
import torch.nn.functional as F


def evaluate_perplexity(model, data_loader, device=None):
    """
    Evaluates Cross-Entropy Loss, Perplexity (PPL), Bits Per Character (BPC),
    and Top-1 Token Accuracy over a PyTorch DataLoader.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0
    total_chars = 0
    correct_tokens = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            bsz, seqlen = x.shape

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            num_toks = y.numel()
            total_tokens += num_toks

            # Calculate Top-1 Token Accuracy
            preds = torch.argmax(logits, dim=-1)
            correct_tokens += (preds == y).sum().item()

            # Approximate character count (assuming avg token length or raw estimate)
            total_chars += num_toks * 4

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    # BPC: Bits per character = (loss_in_nats / ln(2)) * (tokens / chars)
    bits_per_token = avg_loss / math.log(2)
    bpc = bits_per_token * (total_tokens / max(total_chars, 1))
    token_accuracy = (correct_tokens / max(total_tokens, 1)) * 100.0

    return {
        "avg_loss": avg_loss,
        "perplexity": ppl,
        "bpc": bpc,
        "bits_per_token": bits_per_token,
        "token_accuracy": token_accuracy,
        "total_tokens": total_tokens,
    }
