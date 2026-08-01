import os
import sys
import click
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from tiny_llm.data import SentencesDataset
from tiny_llm.models.factory import MODEL_REGISTRY, create_model
from tiny_llm.modules.lora import inject_lora

console = Console()


@click.command("train")
@click.option(
    "--arch",
    type=click.Choice(list(MODEL_REGISTRY.keys()), case_sensitive=False),
    default="dense",
    help="Model architecture (dense, moe, nano, bitnet). Default: dense.",
)
@click.option(
    "--data",
    default="data/corpus.txt",
    help="Path to corpus text file. Default: data/corpus.txt",
)
@click.option(
    "--tokenizer-path",
    default="checkpoints/tokenizer.json",
    help="Path to trained tokenizer.json. Default: checkpoints/tokenizer.json",
)
@click.option(
    "--use-scratch-tokenizer",
    is_flag=True,
    help="Use educational ScratchTokenizer instead of HuggingFace Tokenizer",
)
@click.option("--epochs", type=int, default=10, help="Number of training epochs")
@click.option("--batch-size", type=int, default=32, help="Training batch size")
@click.option("--lr", type=float, default=3e-4, help="Learning rate")
@click.option("--dim", type=int, default=128, help="Embedding dimension size")
@click.option("--n-layers", type=int, default=4, help="Number of transformer layers")
@click.option("--n-heads", type=int, default=4, help="Number of attention heads")
@click.option("--ffn-dim", type=int, default=512, help="Feed-forward network hidden dimension")
@click.option("--max-seq-len", type=int, default=64, help="Maximum sequence length")
@click.option("--n-kv-heads", type=int, default=2, help="KV heads count for GQA (MoE model)")
@click.option("--num-experts", type=int, default=8, help="Total experts count (MoE model)")
@click.option("--num-experts-per-tok", type=int, default=2, help="Top-K experts per token (MoE model)")
@click.option("--checkpoint-dir", default="checkpoints", help="Directory to save model checkpoints")
@click.option("--lora", is_flag=True, help="Inject LoRA adapters for parameter-efficient tuning")
@click.option("--lora-rank", type=int, default=8, help="LoRA rank dimension")
@click.option("--lora-alpha", type=int, default=16, help="LoRA scaling alpha")
def train_cmd(
    arch,
    data,
    tokenizer_path,
    use_scratch_tokenizer,
    epochs,
    batch_size,
    lr,
    dim,
    n_layers,
    n_heads,
    ffn_dim,
    max_seq_len,
    n_kv_heads,
    num_experts,
    num_experts_per_tok,
    checkpoint_dir,
    lora,
    lora_rank,
    lora_alpha,
):
    """🏋️ Train a custom LLM model architecture."""
    console.print(
        Panel.fit(
            f"[bold green]Initializing Training Pipeline[/bold green]\n"
            f"Architecture: [bold cyan]{arch.upper()}[/bold cyan] | Device: [bold yellow]{'CUDA' if torch.cuda.is_available() else 'CPU'}[/bold yellow]",
            title="TinyLLM Trainer",
            border_style="cyan",
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Tokenizer
    if not os.path.exists(tokenizer_path):
        if os.path.exists("tokenizer.json"):
            tokenizer_path = "tokenizer.json"
        else:
            console.print(f"[bold red]Error:[/bold red] Tokenizer not found at '{tokenizer_path}'.")
            console.print("Please run [bold cyan]tiny-llm prepare-data[/bold cyan] first.")
            sys.exit(1)

    if use_scratch_tokenizer:
        from tiny_llm.tokenizer import ScratchTokenizer

        tokenizer = ScratchTokenizer.from_file(tokenizer_path)
        vocab_size = len(tokenizer.vocab)
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()

    # 2. Check Dataset
    if not os.path.exists(data):
        console.print(f"[bold red]Error:[/bold red] Training corpus not found at '{data}'.")
        sys.exit(1)

    dataset = SentencesDataset(data, tokenizer_path, max_length=max_seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Create Model & Config
    model, config = create_model(
        arch=arch,
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
        n_kv_heads=n_kv_heads,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    if lora:
        console.print(f"[bold yellow]Injecting LoRA adapters (rank={lora_rank}, alpha={lora_alpha})...[/bold yellow]")
        inject_lora(model, r=lora_rank, alpha=lora_alpha)

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Display Model Parameters Table
    table = Table(title="Model Hyperparameters", border_style="dim")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Architecture", arch.upper())
    table.add_row("Trainable Parameters", f"{total_params:,}")
    table.add_row("Vocabulary Size", str(vocab_size))
    table.add_row("Embedding Dimension", str(dim))
    table.add_row("Transformer Layers", str(n_layers))
    table.add_row("Attention Heads", str(n_heads))
    table.add_row("FFN Hidden Dimension", str(ffn_dim))

    if arch == "moe":
        table.add_row("Num Experts", str(num_experts))
        table.add_row("Top-K Experts/Tok", str(num_experts_per_tok))

    console.print(table)

    # 4. Training Loop with Rich Progress
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    os.makedirs(checkpoint_dir, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        epoch_task = progress.add_task("[cyan]Training Epochs...", total=epochs)

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for step, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(dataloader))
            progress.update(
                epoch_task,
                advance=1,
                description=f"[cyan]Epoch [{epoch}/{epochs}] | Loss: [bold yellow]{avg_loss:.4f}[/bold yellow]",
            )

    # 5. Save Checkpoint & Config JSON
    ckpt_name = f"{arch}_model.pth" if arch != "dense" else "tiny_llm.pth"
    model_path = os.path.join(checkpoint_dir, ckpt_name)
    config_path = os.path.join(checkpoint_dir, "config.json")

    torch.save(model.state_dict(), model_path)
    config.save_json(config_path)

    console.print(f"[bold green]✨ Training completed successfully![/bold green]")
    console.print(f"📦 Model saved to: [bold white]{model_path}[/bold white]")
    console.print(f"⚙️ Config saved to: [bold white]{config_path}[/bold white]")
