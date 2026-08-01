import os
import sys

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.command("export")
@click.option(
    "--checkpoint",
    default="checkpoints/tiny_llm.pth",
    help="Input PyTorch checkpoint path",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["onnx", "c", "q8", "bitnet"], case_sensitive=False),
    default="onnx",
    help="Export target format (onnx, c, q8, bitnet)",
)
@click.option(
    "--output-dir",
    default="ui/assets",
    help="Target output directory for exported assets",
)
def export_cmd(checkpoint, fmt, output_dir):
    """📦 Export model checkpoint to ONNX, C binary, or Quantized formats."""
    fmt = fmt.lower()
    console.print(
        Panel.fit(
            f"[bold green]Exporting Model Weights[/bold green]\n"
            f"Checkpoint: [cyan]{checkpoint}[/cyan] | Format: [magenta]{fmt.upper()}[/magenta]",
            title="TinyLLM Exporter",
            border_style="cyan",
        )
    )

    if not os.path.exists(checkpoint):
        console.print(f"[bold red]Error:[/bold red] Checkpoint not found at '{checkpoint}'.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    with console.status(f"[bold green]Exporting to {fmt.upper()}..."):
        if fmt == "onnx":
            from tools.export.export_onnx import export_onnx

            export_onnx(model_path=checkpoint, output_dir=output_dir)
        elif fmt == "c":
            from tools.export.export_c import export_c

            export_c(
                model_path=checkpoint,
                output_path=os.path.join(output_dir, "model.bin"),
            )
        elif fmt == "q8":
            from tools.export.export_q8 import export_q8

            export_q8(
                model_path=checkpoint,
                output_path=os.path.join(output_dir, "model_q8.bin"),
            )
        elif fmt == "bitnet":
            from tools.export.export_bitnet import export_bitnet

            export_bitnet(
                model_path=checkpoint,
                output_path=os.path.join(output_dir, "model_bitnet.bin"),
            )

    console.print(f"✨ [bold green]Model exported successfully to '{output_dir}'![/bold green]")
