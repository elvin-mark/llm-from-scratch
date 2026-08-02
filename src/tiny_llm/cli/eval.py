import os
import sys

import click
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from torch.utils.data import DataLoader

from tiny_llm.eval import evaluate_perplexity
from tiny_llm.models.factory import load_model_from_checkpoint

console = Console()


def get_quality_badge(ppl: float) -> str:
    """Return colorized quality assessment badge based on Perplexity."""
    if ppl < 15.0:
        return "[bold white on green] 🏆 EXCELLENT (PPL < 15) [/bold white on green]"
    elif ppl < 35.0:
        return "[bold white on blue] ✨ GOOD (PPL 15-35) [/bold white on blue]"
    elif ppl < 75.0:
        return "[bold black on yellow] ⚠️ FAIR (PPL 35-75) [/bold black on yellow]"
    else:
        return "[bold white on red] 🚨 POOR (PPL > 75) [/bold white on red]"


@click.command("eval")
@click.option(
    "--checkpoint",
    default=None,
    help="Path to trained model checkpoint (.pth). Auto-detects if omitted.",
)
@click.option(
    "--dataset",
    default="data/corpus.txt",
    help="Path to validation/test text corpus file",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Evaluation batch size. Default: 16",
)
@click.option(
    "--max-samples",
    type=int,
    default=1000,
    help="Maximum dataset sequences to evaluate. Default: 1000",
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
def eval_cmd(
    checkpoint,
    dataset,
    batch_size,
    max_samples,
    tokenizer_path,
    use_scratch_tokenizer,
):
    """📊 Evaluate dataset Perplexity (PPL), Cross-Entropy Loss, and BPC across checkpoints."""
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

    if not os.path.exists(dataset):
        console.print(f"[bold red]Error:[/bold red] Dataset file not found at '{dataset}'.")
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
    max_seq_len = getattr(config, "max_seq_len", 64)

    from tiny_llm.data import SentencesDataset

    test_dataset = SentencesDataset(dataset, tokenizer_path, max_length=max_seq_len)
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    console.print(
        Panel.fit(
            f"[bold green]Dataset Perplexity & Quality Evaluator[/bold green]\n"
            f"Checkpoint: [cyan]{checkpoint_path}[/cyan] ({arch})\n"
            f"Dataset: [yellow]{dataset}[/yellow] | Sequences: [magenta]{len(test_dataset)}[/magenta]",
            title="TinyLLM Eval",
            border_style="cyan",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task(description="[bold green]Computing Cross-Entropy Loss...", total=None)
        metrics = evaluate_perplexity(model, data_loader, device=device)

    # Render Evaluation Summary Table
    table = Table(title="📊 Quantitative Model Quality Report", border_style="dim")
    table.add_column("Evaluation Metric", style="cyan")
    table.add_column("Score / Value", style="bold yellow", justify="right")
    table.add_column("Unit / Description", style="magenta")

    table.add_row(
        "Cross-Entropy Loss",
        f"{metrics['avg_loss']:.4f}",
        "nats (natural logarithm loss)",
    )
    table.add_row(
        "Perplexity (PPL)",
        f"{metrics['perplexity']:.2f}",
        "exp(loss) lower is better",
    )
    table.add_row(
        "Bits Per Token (BPT)",
        f"{metrics['bits_per_token']:.3f}",
        "bits of entropy per subword token",
    )
    table.add_row(
        "Bits Per Character (BPC)",
        f"{metrics['bpc']:.3f}",
        "bits of entropy per character",
    )
    table.add_row(
        "Top-1 Token Accuracy",
        f"{metrics['token_accuracy']:.2f}%",
        "exact next-token match rate",
    )
    table.add_row(
        "Total Tokens Evaluated",
        f"{metrics['total_tokens']:,}",
        "subword tokens processed",
    )

    console.print(table)
    console.print(f"Quality Rating: {get_quality_badge(metrics['perplexity'])}\n")
