import os
import sys
import click
from rich.console import Console
from rich.panel import Panel

from tiny_llm.data import prepare_and_train_tokenizer

console = Console()


@click.command("prepare-data")
@click.option(
    "--input",
    "input_path",
    required=True,
    help="Path to input text or TSV file",
)
@click.option(
    "--corpus-file", default="data/corpus.txt", help="Path to output corpus file"
)
@click.option(
    "--tokenizer-out",
    default="checkpoints/tokenizer.json",
    help="Path to output tokenizer JSON",
)
@click.option(
    "--vocab-size", type=int, default=4000, help="Target BPE vocabulary size"
)
@click.option(
    "--scratch-tokenizer",
    is_flag=True,
    help="Train using from-scratch Python tokenizer",
)
def prepare_data_cmd(
    input_path, corpus_file, tokenizer_out, vocab_size, scratch_tokenizer
):
    """🔤 Process raw dataset and train BPE tokenizer."""
    if not os.path.exists(input_path):
        console.print(
            f"[bold red]Error:[/bold red] Input file '{input_path}' does not exist."
        )
        sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold green]Data & Tokenizer Preparation[/bold green]\n"
            f"Input: [cyan]{input_path}[/cyan] | Target Vocab Size: [magenta]{vocab_size}[/magenta]",
            title="TinyLLM Data Pipeline",
            border_style="cyan",
        )
    )

    with console.status("[bold green]Extracting sentences & training BPE tokenizer..."):
        prepare_and_train_tokenizer(
            input_file=input_path,
            corpus_file=corpus_file,
            vocab_size=vocab_size,
            use_scratch_tokenizer=scratch_tokenizer,
            tokenizer_out=tokenizer_out,
        )

    console.print("✨ [bold green]Data preparation and tokenizer training completed![/bold green]")
    console.print(f"📄 Corpus file: [bold white]{corpus_file}[/bold white]")
    console.print(f"🔤 Tokenizer JSON: [bold white]{tokenizer_out}[/bold white]")
