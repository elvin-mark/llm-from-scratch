import os
import sys

import click
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tiny_llm.models.factory import load_model_from_checkpoint

console = Console()


def get_color_styled_val(val: float) -> str:
    """Format attention probability into colorized Rich string."""
    if val >= 0.4:
        return f"[bold white on green] {val:.2f} [/bold white on green]"
    elif val >= 0.2:
        return f"[bold white on blue] {val:.2f} [/bold white on blue]"
    elif val >= 0.08:
        return f"[bold yellow] {val:.2f} [/bold yellow]"
    elif val >= 0.02:
        return f"[cyan] {val:.2f} [/cyan]"
    else:
        return "[dim] .00 [/dim]"


@click.command("viz-attn")
@click.option(
    "--checkpoint",
    default=None,
    help="Path to trained model checkpoint (.pth)",
)
@click.option(
    "--prompt",
    default="The cat sat on the mat",
    help="Input prompt to compute attention weights for",
)
@click.option(
    "--layer",
    type=int,
    default=0,
    help="Layer index to visualize (0-indexed)",
)
@click.option(
    "--head",
    type=int,
    default=0,
    help="Attention head index to visualize (0-indexed)",
)
@click.option(
    "--tokenizer-path",
    default="checkpoints/tokenizer.json",
    help="Path to tokenizer configuration JSON",
)
@click.option(
    "--use-scratch-tokenizer",
    is_flag=True,
    help="Use educational ScratchTokenizer",
)
def viz_attn_cmd(
    checkpoint,
    prompt,
    layer,
    head,
    tokenizer_path,
    use_scratch_tokenizer,
):
    """🎨 Visualize attention weight heatmaps (A_ij) in terminal across heads and layers."""
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    n_layers = getattr(config, "n_layers", len(model.layers))
    n_heads = getattr(config, "n_heads", 4)

    if layer >= n_layers or layer < 0:
        console.print(
            f"[bold red]Error:[/bold red] Layer index {layer} out of bounds (model has {n_layers} layers)."
        )
        sys.exit(1)

    if head >= n_heads or head < 0:
        console.print(
            f"[bold red]Error:[/bold red] Head index {head} out of bounds (model has {n_heads} heads)."
        )
        sys.exit(1)

    # Load Tokenizer
    if not os.path.exists(tokenizer_path):
        if os.path.exists("checkpoints/tokenizer.json"):
            tokenizer_path = "checkpoints/tokenizer.json"
        elif os.path.exists("tokenizer.json"):
            tokenizer_path = "tokenizer.json"

    if use_scratch_tokenizer:
        from tiny_llm.tokenizer import ScratchTokenizer

        tokenizer = ScratchTokenizer.from_file(tokenizer_path)
        encoded = tokenizer.encode(prompt)
        enc_ids = encoded.ids if hasattr(encoded, "ids") else encoded
        token_ids = [tokenizer.vocab.get("[CLS]", 1)] + enc_ids
        inv_map = getattr(tokenizer, "inv_vocab", getattr(tokenizer, "vocab_inv", {}))
        tokens_text = ["[CLS]"] + [inv_map.get(tid, str(tid)) for tid in token_ids[1:]]
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        encoded = tokenizer.encode(prompt)
        cls_id = tokenizer.token_to_id("[CLS]")
        token_ids = [cls_id] + encoded.ids if cls_id is not None else encoded.ids
        tokens_text = ["[CLS]"] + encoded.tokens if cls_id is not None else encoded.tokens

    tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Forward pass with attention weight extraction
    with torch.no_grad():
        try:
            logits, all_weights = model(tokens_tensor, return_attn_weights=True)
        except TypeError:
            console.print(
                "[bold yellow]Warning:[/bold yellow] Attention weight hook not active for this model architecture."
            )
            sys.exit(1)

    # Extract 2D matrix A[i, j] for requested layer and head
    attn_matrix = all_weights[layer][0, head].cpu()  # shape: [seqlen, seqlen]
    seqlen = len(token_ids)

    console.print(
        Panel.fit(
            f"[bold green]Attention Heatmap Visualizer[/bold green]\n"
            f'Prompt: [bold cyan]"{prompt}"[/bold cyan]\n'
            f"Layer: [bold magenta]{layer}/{n_layers - 1}[/bold magenta] | Head: [bold magenta]{head}/{n_heads - 1}[/bold magenta] | Sequence Length: [yellow]{seqlen}[/yellow]",
            title="TinyLLM VizAttn",
            border_style="cyan",
        )
    )

    # Build Rich Attention Heatmap Table
    matrix_table = Table(
        title=f"Attention Weights A_ij (Layer {layer}, Head {head})",
        border_style="dim",
        show_lines=True,
    )

    matrix_table.add_column("Query \\ Key", style="bold cyan", justify="right")
    for t_str in tokens_text[:seqlen]:
        matrix_table.add_column(t_str, justify="center")

    top_pairs = []

    for i in range(seqlen):
        row_cells = [tokens_text[i]]
        for j in range(seqlen):
            prob = attn_matrix[i, j].item()
            row_cells.append(get_color_styled_val(prob))
            if i != j:  # Exclude self-attention for top pairs report
                top_pairs.append((prob, tokens_text[i], tokens_text[j]))

        matrix_table.add_row(*row_cells)

    console.print(matrix_table)

    # Display Top Attended Token Pairs Breakdown
    top_pairs.sort(key=lambda x: x[0], reverse=True)
    summary_table = Table(title="🔥 Top Attended Token Pairs (Excluding Self)", border_style="dim")
    summary_table.add_column("Rank", style="dim", justify="center")
    summary_table.add_column("Query Token (i)", style="cyan")
    summary_table.add_column("Attends To (Key j)", style="green")
    summary_table.add_column("Attention Weight", style="bold yellow", justify="right")

    for rank, (prob, q_tok, k_tok) in enumerate(top_pairs[:5], 1):
        summary_table.add_row(str(rank), q_tok, k_tok, f"{prob * 100:.1f}%")

    console.print(summary_table)
