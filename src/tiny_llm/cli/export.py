import os
import sys

import click
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from tiny_llm.models.factory import load_model_from_checkpoint

console = Console()


def format_bytes(num_bytes: int) -> str:
    """Format bytes count into human-readable string."""
    if num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    else:
        return f"{num_bytes} B"


@click.command("export")
@click.option(
    "--checkpoint",
    default=None,
    help="Path to trained PyTorch checkpoint (.pth). Auto-detects if omitted.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(
        ["onnx", "c", "bin", "binary", "q8", "quantized", "bitnet"], case_sensitive=False
    ),
    default="onnx",
    help="Target export format (onnx, c/bin, q8, bitnet). Default: onnx",
)
@click.option(
    "--output-dir",
    default="ui/assets",
    help="Target output directory for exported assets. Default: ui/assets",
)
@click.option(
    "--tokenizer-path",
    default="checkpoints/tokenizer.json",
    help="Path to tokenizer JSON file",
)
@click.option(
    "--quantize",
    is_flag=True,
    help="Enable Int8 quantization for ONNX export",
)
def export_cmd(checkpoint, fmt, output_dir, tokenizer_path, quantize):
    """📦 Export PyTorch checkpoints to ONNX, bare-metal C binary, or Int8 Quantized formats."""
    fmt = fmt.lower()
    if fmt in ("c", "bin", "binary"):
        fmt = "c"
    elif fmt in ("q8", "quantized"):
        fmt = "q8"

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

    os.makedirs(output_dir, exist_ok=True)

    console.print(
        Panel.fit(
            f"[bold green]Exporting Model Weights[/bold green]\n"
            f"Source Checkpoint: [cyan]{checkpoint_path}[/cyan]\n"
            f"Target Format: [bold magenta]{fmt.upper()}[/bold magenta] | Output Directory: [yellow]{output_dir}[/yellow]",
            title="TinyLLM Exporter",
            border_style="cyan",
        )
    )

    exported_files = []

    with console.status(f"[bold green]Exporting weights to {fmt.upper()} format..."):
        if fmt == "onnx":
            from tools.export.export_onnx import export_to_onnx

            out_name = "tiny_llm_quant.onnx" if quantize else "tiny_llm.onnx"
            out_path = os.path.join(output_dir, out_name)
            export_to_onnx(
                model_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                output_path=out_path,
                quantize=quantize,
            )
            exported_files.append((out_path, "ONNX Computational Graph (Browser / Serverless)"))

        elif fmt == "c":
            from tools.export.export_c import export_model

            out_path = os.path.join(output_dir, "model.bin")
            vocab_out = os.path.join(output_dir, "vocab.bin")
            export_model(
                model_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                output_path=out_path,
                vocab_path=vocab_out,
            )
            exported_files.append((out_path, "Bare-metal FP32 C Engine Weights Binary"))
            if os.path.exists(vocab_out):
                exported_files.append((vocab_out, "C Engine Tokenizer Vocabulary Binary"))

        elif fmt == "q8":
            from tools.export.export_q8 import export_model_q8

            out_path = os.path.join(output_dir, "model_q8.bin")
            export_model_q8(
                model_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                output_path=out_path,
            )
            exported_files.append((out_path, "Row-wise Int8 Dynamic Quantized C Binary"))

        elif fmt == "bitnet":
            from tools.export.export_bitnet import export_bitnet

            out_path = os.path.join(output_dir, "model_bitnet.bin")
            export_bitnet(
                model_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                output_path=out_path,
            )
            exported_files.append((out_path, "BitNet 1.58-Bit Packed Ternary Binary"))

    # Render Summary Table
    table = Table(title="✨ Exported Artifacts Summary", border_style="dim")
    table.add_column("File Path", style="cyan")
    table.add_column("Format Description", style="magenta")
    table.add_column("File Size", style="bold green", justify="right")

    for path, desc in exported_files:
        size_str = format_bytes(os.path.getsize(path)) if os.path.exists(path) else "N/A"
        table.add_row(path, desc, size_str)

    console.print(table)
    console.print("[bold green]✨ Export completed successfully![/bold green]")
