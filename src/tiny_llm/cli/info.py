import os
import sys

import click
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tiny_llm.models.factory import load_model_from_checkpoint

console = Console()


def format_bytes(num_bytes: int) -> str:
    """Format bytes count into human-readable MB / KB / GB string."""
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f} GB"
    elif num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    else:
        return f"{num_bytes} Bytes"


@click.command("info")
@click.option(
    "--checkpoint",
    default=None,
    help="Path to trained model checkpoint (.pth)",
)
def info_cmd(checkpoint):
    """🔍 Inspect model checkpoint architecture, parameter breakdown, and memory footprint."""
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

    arch = getattr(config, "arch", "dense").upper()
    vocab_size = getattr(config, "vocab_size", 4000)
    dim = getattr(config, "dim", 128)
    n_layers = getattr(config, "n_layers", len(model.layers))
    n_heads = getattr(config, "n_heads", 4)
    ffn_dim = getattr(config, "ffn_dim", 512)
    max_seq_len = getattr(config, "max_seq_len", 128)

    # Calculate parameter counts per component group
    embed_params = sum(p.numel() for p in model.tok_embeddings.parameters())
    output_params = sum(p.numel() for p in model.output.parameters())

    attn_params = 0
    ffn_params = 0
    norm_params = sum(p.numel() for p in model.norm.parameters())

    for layer in model.layers:
        attn_params += sum(p.numel() for p in layer.attention.parameters())
        ffn_params += sum(p.numel() for p in layer.feed_forward.parameters())
        if hasattr(layer, "attention_norm"):
            norm_params += sum(p.numel() for p in layer.attention_norm.parameters())
        if hasattr(layer, "ffn_norm"):
            norm_params += sum(p.numel() for p in layer.ffn_norm.parameters())

    # Handle NanoLLM weight tying
    if arch == "NANO":
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = embed_params + output_params + attn_params + ffn_params + norm_params

    console.print(
        Panel.fit(
            f"[bold green]Model Checkpoint Inspector[/bold green]\n"
            f"Path: [cyan]{checkpoint_path}[/cyan]\n"
            f"Architecture: [bold cyan]{arch}[/bold cyan] | Total Parameters: [bold yellow]{total_params:,}[/bold yellow]",
            title="TinyLLM Info",
            border_style="cyan",
        )
    )

    # 1. Architecture Specs Table
    specs_table = Table(title="🏗️ Architecture & Hyperparameters", border_style="dim")
    specs_table.add_column("Property", style="cyan")
    specs_table.add_column("Value", style="magenta")

    specs_table.add_row("Architecture Type", arch)
    specs_table.add_row("Vocabulary Size (V)", f"{vocab_size:,}")
    specs_table.add_row("Embedding Dimension (d)", str(dim))
    specs_table.add_row("Transformer Layers (L)", str(n_layers))
    specs_table.add_row("Attention Heads (n_heads)", str(n_heads))
    specs_table.add_row("Head Dimension (d_head)", str(dim // n_heads))

    if hasattr(config, "n_kv_heads"):
        n_kv_heads = config.n_kv_heads
        gqa_ratio = n_heads // n_kv_heads
        specs_table.add_row("KV Heads (GQA)", f"{n_kv_heads} (GQA ratio 1:{gqa_ratio})")

    specs_table.add_row("FFN Hidden Dim (d_ffn)", str(ffn_dim))
    specs_table.add_row("Context Window (max_seq_len)", f"{max_seq_len} tokens")

    if hasattr(config, "num_experts"):
        specs_table.add_row("MoE Experts Count", str(config.num_experts))
        specs_table.add_row("MoE Top-K Experts/Tok", str(config.num_experts_per_tok))

    console.print(specs_table)

    # 2. Parameter Breakdown Table
    param_table = Table(title="🧩 Parameter Count Breakdown", border_style="dim")
    param_table.add_column("Component", style="cyan")
    param_table.add_column("Parameters", style="bold yellow", justify="right")
    param_table.add_column("Percentage", style="green", justify="right")

    components = [
        ("Token Embeddings", embed_params),
        ("Attention Blocks", attn_params),
        ("Feed-Forward Networks", ffn_params),
        ("Layer Normalization", norm_params),
        ("Output Projection Head", output_params if arch != "NANO" else 0),
    ]

    for comp_name, count in components:
        pct = (count / max(total_params, 1)) * 100
        param_table.add_row(comp_name, f"{count:,}", f"{pct:.1f}%")

    param_table.add_section()
    param_table.add_row("[bold]Total Trainable[/bold]", f"[bold]{total_params:,}[/bold]", "100.0%")
    console.print(param_table)

    # 3. Estimated Memory Footprint Table
    mem_table = Table(title="💾 Estimated Memory Footprint (Weights Only)", border_style="dim")
    mem_table.add_column("Precision Format", style="cyan")
    mem_table.add_column("Bits per Param", style="magenta", justify="center")
    mem_table.add_column("Estimated Memory", style="bold green", justify="right")

    fp32_bytes = total_params * 4
    fp16_bytes = total_params * 2
    int8_bytes = total_params * 1
    bitnet_bytes = int(total_params * (1.58 / 8))

    mem_table.add_row("FP32 (Standard Float32)", "32 bits", format_bytes(fp32_bytes))
    mem_table.add_row("FP16 / BF16 (Half Precision)", "16 bits", format_bytes(fp16_bytes))
    mem_table.add_row("Int8 / Q8 (Dynamic Quantized)", "8 bits", format_bytes(int8_bytes))
    mem_table.add_row("BitNet 1.58-Bit (Ternary {-1,0,1})", "1.58 bits", format_bytes(bitnet_bytes))

    console.print(mem_table)
