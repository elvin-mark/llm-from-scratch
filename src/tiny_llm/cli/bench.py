import sys

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.command("bench")
@click.option(
    "--suite",
    type=click.Choice(
        ["attention", "flash_attn", "moe", "kv_cache", "bitnet"], case_sensitive=False
    ),
    default="attention",
    help="Benchmark suite to run",
)
def bench_cmd(suite):
    """⚡ Run benchmark performance suite on model components."""
    suite = suite.lower()
    console.print(
        Panel.fit(
            f"[bold green]Running Benchmark Suite[/bold green]\n"
            f"Suite: [bold cyan]{suite.upper()}[/bold cyan]",
            title="TinyLLM Benchmarks",
            border_style="cyan",
        )
    )

    if suite == "attention":
        from experiments.bench_attention import run_benchmark

        run_benchmark()
    elif suite == "flash_attn":
        from experiments.bench_flash_attn import run_benchmark

        run_benchmark()
    elif suite == "moe":
        from experiments.bench_moe import run_benchmark

        run_benchmark()
    elif suite == "kv_cache":
        from experiments.bench_kv_cache import run_benchmark

        run_benchmark()
    elif suite == "bitnet":
        from experiments.bench_bitnet import run_benchmark

        run_benchmark()
    else:
        console.print(f"[bold red]Error:[/bold red] Unknown suite '{suite}'.")
        sys.exit(1)
