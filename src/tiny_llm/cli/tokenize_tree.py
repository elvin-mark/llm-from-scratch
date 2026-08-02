import os
import re
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


@click.command("tokenize-tree")
@click.option(
    "--text",
    "-t",
    default="Unbelievable processing of subwords!",
    help="Text string to tokenize and visualize as a BPE subword tree",
)
@click.option(
    "--tokenizer-path",
    default="checkpoints/tokenizer.json",
    help="Path to tokenizer configuration JSON file",
)
@click.option(
    "--use-scratch-tokenizer",
    is_flag=True,
    help="Use educational pure-Python ScratchTokenizer",
)
def tokenize_tree_cmd(text, tokenizer_path, use_scratch_tokenizer):
    """🌲 Renders an interactive ASCII BPE subword tree and token ID breakdown."""
    if not os.path.exists(tokenizer_path):
        if os.path.exists("checkpoints/tokenizer.json"):
            tokenizer_path = "checkpoints/tokenizer.json"
        elif os.path.exists("tokenizer.json"):
            tokenizer_path = "tokenizer.json"
        else:
            console.print(
                f"[bold red]Error:[/bold red] Tokenizer configuration file not found at '{tokenizer_path}'."
            )
            sys.exit(1)

    # Load Tokenizer
    if use_scratch_tokenizer:
        from tiny_llm.tokenizer import ScratchTokenizer

        tokenizer = ScratchTokenizer.from_file(tokenizer_path)
        encoded = tokenizer.encode(text)
        token_ids = encoded.ids if hasattr(encoded, "ids") else encoded
        inv_map = getattr(tokenizer, "inv_vocab", getattr(tokenizer, "vocab_inv", {}))
        token_strings = [inv_map.get(tid, str(tid)) for tid in token_ids]
        tok_type_label = "Educational ScratchTokenizer"
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        encoded = tokenizer.encode(text)
        token_ids = encoded.ids
        token_strings = encoded.tokens
        tok_type_label = "Hugging Face BPE Tokenizer"

    num_tokens = len(token_ids)
    num_chars = len(text)
    ratio = num_chars / max(num_tokens, 1)

    console.print(
        Panel.fit(
            f"[bold green]BPE Subword Tokenizer Tree Visualizer[/bold green]\n"
            f'Input Text: [bold cyan]"{text}"[/bold cyan]\n'
            f"Tokenizer: [magenta]{tok_type_label}[/magenta] ({tokenizer_path})\n"
            f"Tokens: [bold yellow]{num_tokens}[/bold yellow] | Characters: [bold yellow]{num_chars}[/bold yellow] | Efficiency: [bold green]{ratio:.2f} chars/token[/bold green]",
            title="TinyLLM TokenizeTree",
            border_style="cyan",
        )
    )

    # Group tokens by word boundaries
    words = re.findall(r"\S+|\s+", text)
    root_tree = Tree(f"🌲 [bold white]Input Sentence[/bold white] ([cyan]{text}[/cyan])")

    # Match tokens to words
    tok_idx = 0
    subwords_by_word = []

    for word in words:
        if not word.strip():
            continue
        word_node = root_tree.add(f'📄 Word: [bold cyan]"{word}"[/bold cyan]')
        word_subwords = []

        # Find tokens belonging to this word
        accumulated = ""
        while tok_idx < num_tokens and len(accumulated) < len(word):
            t_str = token_strings[tok_idx]
            t_id = token_ids[tok_idx]

            # Clean subword representation
            clean_str = t_str.replace("Ġ", " ").replace("##", "").replace(" ", " ")
            accumulated += clean_str.strip()

            char_len = len(clean_str.strip())
            tok_type = (
                "[bold magenta]Special[/bold magenta]"
                if t_str.startswith("[") and t_str.endswith("]")
                else ("[cyan]Subword[/cyan]" if char_len > 1 else "[yellow]Char/Byte[/yellow]")
            )

            word_node.add(
                f'├── [bold yellow]"{clean_str}"[/bold yellow]  '
                f"[dim]ID:[/dim] [bold green]{t_id}[/bold green]  "
                f"[dim]Length:[/dim] {char_len} chars  "
                f"[dim]Type:[/dim] {tok_type}"
            )
            word_subwords.append((t_str, clean_str, t_id, char_len))
            tok_idx += 1

        subwords_by_word.append((word, word_subwords))

    # Catch remaining tokens
    if tok_idx < num_tokens:
        rem_node = root_tree.add("📄 [dim]Trailing / Special Tokens[/dim]")
        while tok_idx < num_tokens:
            t_str = token_strings[tok_idx]
            t_id = token_ids[tok_idx]
            rem_node.add(
                f'├── [bold yellow]"{t_str}"[/bold yellow]  [dim]ID:[/dim] [bold green]{t_id}[/bold green]'
            )
            tok_idx += 1

    console.print(root_tree)

    # Render Detailed Token Breakdown Table
    colors = ["cyan", "magenta", "yellow", "green", "blue"]
    table = Table(title="📊 Sequential Token Decomposition Table", border_style="dim")
    table.add_column("Index", style="dim", justify="center")
    table.add_column("Token String", justify="center")
    table.add_column("Token ID", style="bold green", justify="right")
    table.add_column("Character Length", style="yellow", justify="right")
    table.add_column("Category", style="magenta")

    for i, (t_str, t_id) in enumerate(zip(token_strings, token_ids)):
        color = colors[i % len(colors)]
        display_str = f'[{color}]"{t_str}"[/{color}]'
        c_len = len(t_str.replace("Ġ", "").replace("##", ""))

        category = (
            "Special Token"
            if t_str.startswith("[") and t_str.endswith("]")
            else ("Subword Segment" if c_len > 1 else "Single Character / Byte")
        )

        table.add_row(str(i), display_str, str(t_id), str(c_len), category)

    console.print(table)
