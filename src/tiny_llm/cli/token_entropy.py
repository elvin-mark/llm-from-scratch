import math
import os
import sys

import click
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tiny_llm.models.factory import load_model_from_checkpoint

console = Console()


def get_styled_entropy_token(token_str: str, entropy: float) -> str:
    """Return Rich markup styled string for a token based on its Entropy (bits)."""
    clean_str = token_str.replace("Ġ", " ").replace("##", "")
    if entropy < 1.2:
        return f"[bold white on green] {clean_str} [/bold white on green]"
    elif entropy <= 3.0:
        return f"[bold black on yellow] {clean_str} [/bold black on yellow]"
    else:
        return f"[bold white on red] {clean_str} [/bold white on red]"


@click.command("token-entropy")
@click.option(
    "--checkpoint",
    default=None,
    help="Path to trained model checkpoint (.pth). Auto-detects if omitted.",
)
@click.option(
    "--prompt",
    "-p",
    default="Once upon a time in a tiny land",
    help="Input prompt to analyze token entropy and surprisal for",
)
@click.option(
    "--tokenizer-path",
    default="checkpoints/tokenizer.json",
    help="Path to tokenizer JSON file",
)
@click.option(
    "--use-scratch-tokenizer",
    is_flag=True,
    help="Use educational ScratchTokenizer",
)
def token_entropy_cmd(checkpoint, prompt, tokenizer_path, use_scratch_tokenizer):
    """🟢 Visualizes Shannon Entropy H(p) and Surprisal I(x) heatmaps per token in terminal."""
    checkpoint_path = checkpoint
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        candidates = [
            "checkpoints/moe_model.pth",
            "checkpoints/bitnet_model.pth",
            "checkpoints/nano_model.pth",
            "checkpoints/tiny_llm.pth",
            ".models/kor/tiny_llm.pth",
            "tiny_llm.pth",
        ]
        found = None
        for cand in candidates:
            if os.path.exists(cand):
                found = cand
                break
        if found:
            checkpoint_path = found
        else:
            console.print(
                f"[bold red]Error:[/bold red] Model checkpoint not found at '{checkpoint_path}'."
            )
            sys.exit(1)

    if not os.path.exists(tokenizer_path):
        if os.path.exists("checkpoints/tokenizer.json"):
            tokenizer_path = "checkpoints/tokenizer.json"
        elif os.path.exists("tokenizer.json"):
            tokenizer_path = "tokenizer.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    arch = getattr(config, "arch", "dense").upper()

    # Load Tokenizer & Encode Prompt
    if use_scratch_tokenizer:
        from tiny_llm.tokenizer import ScratchTokenizer

        tokenizer = ScratchTokenizer.from_file(tokenizer_path)
        encoded = tokenizer.encode(prompt)
        enc_ids = encoded.ids if hasattr(encoded, "ids") else encoded
        cls_id = tokenizer.vocab.get("[CLS]", 1)
        token_ids = [cls_id] + enc_ids
        inv_map = getattr(tokenizer, "inv_vocab", getattr(tokenizer, "vocab_inv", {}))
        token_strings = ["[CLS]"] + [inv_map.get(tid, str(tid)) for tid in enc_ids]
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        encoded = tokenizer.encode(prompt)
        cls_id = tokenizer.token_to_id("[CLS]")
        token_ids = [cls_id] + encoded.ids if cls_id is not None else encoded.ids
        token_strings = ["[CLS]"] + encoded.tokens if cls_id is not None else encoded.tokens

    tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    seqlen = len(token_ids)

    # Perform Forward Pass
    with torch.no_grad():
        logits = model(tokens_tensor)  # shape: [1, seqlen, vocab_size]
        probs = F.softmax(logits[0], dim=-1)  # shape: [seqlen, vocab_size]
        log_probs_base2 = F.log_softmax(logits[0], dim=-1) / math.log(2)

        # 1. Shannon Entropy H(p) = -sum(p * log2(p))
        entropies = -torch.sum(probs * log_probs_base2, dim=-1).cpu()  # shape: [seqlen]

        # 2. Surprisal I(x_t) = -log2 P(x_t)
        targets_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        surprisals = -log_probs_base2.gather(-1, targets_tensor.unsqueeze(-1)).squeeze(-1).cpu()

        # Top candidate predictions
        top_probs, top_indices = torch.topk(probs, k=2, dim=-1)

    avg_entropy = torch.mean(entropies).item()
    max_surprisal_idx = torch.argmax(surprisals).item()

    console.print(
        Panel.fit(
            f"[bold green]Token Entropy & Surprisal Heatmap Visualizer[/bold green]\n"
            f'Prompt: [bold cyan]"{prompt}"[/bold cyan]\n'
            f"Model: [cyan]{checkpoint_path}[/cyan] ({arch}) | Tokens: [yellow]{seqlen}[/yellow]\n"
            f'Average Entropy: [bold green]{avg_entropy:.2f} bits[/bold green] | Max Surprisal Token: [bold yellow]"{token_strings[max_surprisal_idx]}" ({surprisals[max_surprisal_idx]:.2f} bits)[/bold yellow]',
            title="TinyLLM TokenEntropy",
            border_style="cyan",
        )
    )

    # 1. Render Colorized Text Stream Block
    console.print("\n[bold white]🔥 Token Confidence Heatmap Stream:[/bold white]")

    for i in range(seqlen):
        t_str = token_strings[i]
        ent = entropies[i].item()
        styled_segment = get_styled_entropy_token(t_str, ent)
        console.print(styled_segment, end="")

    console.print("\n")
    console.print(
        "Legend: [bold white on green] Low Entropy (< 1.2b) [/bold white on green] "
        "[bold black on yellow] Moderate (1.2-3.0b) [/bold black on yellow] "
        "[bold white on red] High Uncertainty (> 3.0b) [/bold white on red]\n"
    )

    # 2. Render Detailed Metrics Table
    table = Table(title="📊 Detailed Token Information-Theoretic Breakdown", border_style="dim")
    table.add_column("Pos", style="dim", justify="center")
    table.add_column("Subword Token", justify="center")
    table.add_column("Token ID", style="bold green", justify="right")
    table.add_column("Prob P(x_t)", style="cyan", justify="right")
    table.add_column("Entropy H(P_t)", style="bold yellow", justify="right")
    table.add_column("Surprisal I(x_t)", style="bold magenta", justify="right")
    table.add_column("Top Candidate (P_top)", style="white")

    for i in range(seqlen):
        t_str = token_strings[i]
        t_id = token_ids[i]
        p_val = probs[i, t_id].item() * 100.0
        h_val = entropies[i].item()
        i_val = surprisals[i].item()

        top_id = top_indices[i, 0].item()
        top_p = top_probs[i, 0].item() * 100.0

        if use_scratch_tokenizer:
            top_str = inv_map.get(top_id, str(top_id))
        else:
            top_str = (
                tokenizer.id_to_token(top_id) if hasattr(tokenizer, "id_to_token") else str(top_id)
            )

        clean_top = str(top_str).replace("Ġ", " ").replace("##", "") if top_str else str(top_id)

        table.add_row(
            str(i),
            f'"{t_str}"',
            str(t_id),
            f"{p_val:.1f}%",
            f"{h_val:.2f} b",
            f"{i_val:.2f} b",
            f'"{clean_top}" ({top_p:.1f}%)',
        )

    console.print(table)
