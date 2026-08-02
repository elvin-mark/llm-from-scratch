import os
import sys

import click
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from tiny_llm.models.factory import load_model_from_checkpoint
from tiny_llm.modules.attention import KVCache

console = Console()


def sample_tokens(
    model,
    input_ids,
    max_tokens=64,
    temperature=0.8,
    top_k=50,
    device="cpu",
    max_seq_len=128,
    use_kv_cache=True,
):
    """Autoregressive text token generator helper supporting KV-caching."""
    model.eval()
    bsz, prompt_len = input_ids.shape
    generated = input_ids.clone().to(device)

    if use_kv_cache:
        n_layers = getattr(model, "n_layers", len(model.layers))
        n_heads = getattr(model, "n_heads", 4)
        dim = getattr(model, "dim", 128)

        first_layer_attn = model.layers[0].attention
        n_kv_heads = getattr(first_layer_attn, "n_kv_heads", n_heads)
        head_dim = dim // n_heads

        kv_caches = [
            KVCache(
                max_batch_size=bsz,
                max_seq_len=prompt_len + max_tokens + 16,
                n_heads=n_kv_heads,
                head_dim=head_dim,
                device=device,
            )
            for _ in range(n_layers)
        ]

        # 1. Prefill Phase (Prompt)
        with torch.no_grad():
            logits = model(generated, start_pos=0, kv_caches=kv_caches)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

        # 2. Decode Phase (Single Token Step)
        start_pos = prompt_len
        curr_token = next_token
        with torch.no_grad():
            for _ in range(max_tokens - 1):
                logits = model(curr_token, start_pos=start_pos, kv_caches=kv_caches)
                next_logits = logits[:, -1, :] / max(temperature, 1e-5)

                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                curr_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, curr_token), dim=1)
                start_pos += 1

    else:
        with torch.no_grad():
            for _ in range(max_tokens):
                cond = generated[:, -max_seq_len:]
                logits = model(cond)
                next_logits = logits[:, -1, :] / max(temperature, 1e-5)

                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

    return generated[0].tolist()


def _run_generate(
    checkpoint,
    tokenizer_path,
    use_scratch_tokenizer,
    prompt,
    interactive,
    max_tokens,
    temperature,
    top_k,
    use_kv_cache=True,
):
    """Internal implementation handler for text generation."""
    checkpoint_path = checkpoint
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        candidates = [
            "checkpoints/moe_model.pth",
            "checkpoints/bitnet_model.pth",
            "checkpoints/nano_model.pth",
            "checkpoints/tiny_llm.pth",
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

    # Auto-detect architecture & config from checkpoint
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)

    console.print(
        Panel.fit(
            f"[bold green]Loaded Model Checkpoint[/bold green]\n"
            f"Architecture: [bold cyan]{config.arch.upper()}[/bold cyan] | Dim: [magenta]{config.dim}[/magenta] | Layers: [magenta]{config.n_layers}[/magenta] | Heads: [magenta]{config.n_heads}[/magenta]\n"
            f"KV-Cache: [bold {'green' if use_kv_cache else 'red'}]{'ENABLED (O(1) decode)' if use_kv_cache else 'DISABLED (O(N^2) re-compute)'}[/bold {'green' if use_kv_cache else 'red'}]",
            title="TinyLLM Generator",
            border_style="cyan",
        )
    )

    # Load Tokenizer
    if not os.path.exists(tokenizer_path):
        if os.path.exists("checkpoints/tokenizer.json"):
            tokenizer_path = "checkpoints/tokenizer.json"
        elif os.path.exists("tokenizer.json"):
            tokenizer_path = "tokenizer.json"

    if use_scratch_tokenizer:
        from tiny_llm.tokenizer import ScratchTokenizer

        tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)

    # Encode Helper
    def encode_text(p):
        if use_scratch_tokenizer:
            cls_id = tokenizer.vocab.get("[CLS]", 1)
            return [cls_id] + tokenizer.encode(p)
        else:
            cls_id = tokenizer.token_to_id("[CLS]")
            ids = tokenizer.encode(p).ids
            return [cls_id] + ids if cls_id is not None else ids

    # Decode Helper
    def decode_ids(ids):
        return tokenizer.decode(ids)

    # Interactive REPL mode
    if interactive:
        console.print(
            "\n💬 [bold green]Entering interactive REPL chat mode[/bold green] (type 'exit' or press Ctrl+C to quit):\n"
        )
        while True:
            try:
                user_prompt = Prompt.ask("[bold cyan]Prompt[/bold cyan]").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ("exit", "quit"):
                    break

                prompt_ids = torch.tensor([encode_text(user_prompt)], dtype=torch.long)
                output_ids = sample_tokens(
                    model,
                    prompt_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=device,
                    max_seq_len=config.max_seq_len,
                    use_kv_cache=use_kv_cache,
                )
                text = decode_ids(output_ids)
                console.print(
                    Panel(
                        text,
                        title="Generated Output",
                        border_style="green",
                        expand=False,
                    )
                )
                console.print()
            except (KeyboardInterrupt, EOFError):
                console.print("\nExiting REPL.")
                break
    else:
        console.print(f"[bold cyan]Prompt:[/bold cyan] '{prompt}'")
        prompt_ids = torch.tensor([encode_text(prompt)], dtype=torch.long)

        output_ids = sample_tokens(
            model,
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            max_seq_len=config.max_seq_len,
            use_kv_cache=use_kv_cache,
        )
        output_text = decode_ids(output_ids)

        console.print(
            Panel(
                output_text,
                title="✨ Generated Output",
                border_style="green",
                expand=False,
            )
        )


@click.command("generate")
@click.option(
    "--checkpoint",
    default=None,
    help="Path to trained model checkpoint (.pth)",
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
@click.option("--prompt", default="Once upon a time", help="Text prompt to initialize generation")
@click.option(
    "-i",
    "--interactive",
    is_flag=True,
    help="Launch interactive REPL mode in terminal",
)
@click.option("--max-tokens", type=int, default=64, help="Maximum number of tokens to generate")
@click.option("--temperature", type=float, default=0.8, help="Sampling temperature")
@click.option("--top-k", type=int, default=50, help="Top-K sampling limit")
@click.option(
    "--use-kv-cache/--no-kv-cache",
    default=True,
    help="Enable or disable stateful KV-Cache for single-token O(1) decoding",
)
def generate_cmd(
    checkpoint,
    tokenizer_path,
    use_scratch_tokenizer,
    prompt,
    interactive,
    max_tokens,
    temperature,
    top_k,
    use_kv_cache,
):
    """🔮 Run autoregressive text generation / inference on any model."""
    _run_generate(
        checkpoint,
        tokenizer_path,
        use_scratch_tokenizer,
        prompt,
        interactive,
        max_tokens,
        temperature,
        top_k,
        use_kv_cache=use_kv_cache,
    )


@click.command("infer")
@click.option(
    "--checkpoint",
    default=None,
    help="Path to trained model checkpoint (.pth)",
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
@click.option("--prompt", default="Once upon a time", help="Text prompt to initialize generation")
@click.option(
    "-i",
    "--interactive",
    is_flag=True,
    help="Launch interactive REPL mode in terminal",
)
@click.option("--max-tokens", type=int, default=64, help="Maximum number of tokens to generate")
@click.option("--temperature", type=float, default=0.8, help="Sampling temperature")
@click.option("--top-k", type=int, default=50, help="Top-K sampling limit")
@click.option(
    "--use-kv-cache/--no-kv-cache",
    default=True,
    help="Enable or disable stateful KV-Cache for single-token O(1) decoding",
)
def infer_cmd(
    checkpoint,
    tokenizer_path,
    use_scratch_tokenizer,
    prompt,
    interactive,
    max_tokens,
    temperature,
    top_k,
    use_kv_cache,
):
    """Alias for 'generate' command."""
    _run_generate(
        checkpoint,
        tokenizer_path,
        use_scratch_tokenizer,
        prompt,
        interactive,
        max_tokens,
        temperature,
        top_k,
        use_kv_cache=use_kv_cache,
    )
