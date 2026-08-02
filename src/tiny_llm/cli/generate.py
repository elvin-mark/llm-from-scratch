import os
import sys
import time

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
    token_callback=None,
):
    """Autoregressive text token generator helper supporting KV-caching and live token callbacks."""
    model.eval()
    bsz, prompt_len = input_ids.shape
    generated = input_ids.clone().to(device)

    t_start = time.perf_counter()

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

            if token_callback is not None:
                token_callback(next_token.item(), time.perf_counter() - t_start)

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

                if token_callback is not None:
                    token_callback(curr_token.item(), time.perf_counter() - t_start)

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

                if token_callback is not None:
                    token_callback(next_token.item(), time.perf_counter() - t_start)

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
    stream=True,
):
    """Internal implementation handler for text generation with live token streaming."""
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

    def run_single_prompt(user_prompt):
        console.print(f"\n[bold cyan]Prompt:[/bold cyan] '{user_prompt}'")
        console.print("[bold green]✨ Output:[/bold green] ", end="")
        console.file.flush()

        prompt_ids = torch.tensor([encode_text(user_prompt)], dtype=torch.long)

        stream_state = {
            "tokens": [],
            "prev_len": 0,
            "t_start": None,
        }

        def stream_callback(token_id, elapsed):
            if stream_state["t_start"] is None:
                stream_state["t_start"] = time.perf_counter()

            stream_state["tokens"].append(token_id)
            if stream:
                full_text = decode_ids(stream_state["tokens"])
                new_text = full_text[stream_state["prev_len"] :]
                stream_state["prev_len"] = len(full_text)
                console.print(new_text, end="")
                console.file.flush()

        t0 = time.perf_counter()
        output_ids = sample_tokens(
            model,
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            max_seq_len=config.max_seq_len,
            use_kv_cache=use_kv_cache,
            token_callback=stream_callback,
        )

        t_total = time.perf_counter() - t0
        num_generated = len(output_ids) - prompt_ids.shape[1]
        tok_per_sec = num_generated / max(t_total, 1e-5)

        if not stream:
            output_text = decode_ids(output_ids)
            console.print(output_text, end="")
            console.file.flush()

        console.print(
            f"\n[dim]⚡ Generated [bold white]{num_generated}[/bold white] tokens in [bold white]{t_total:.2f}s[/bold white] ([bold cyan]{tok_per_sec:.1f} tok/s[/bold cyan])[/dim]\n"
        )

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

                run_single_prompt(user_prompt)
            except (KeyboardInterrupt, EOFError):
                console.print("\nExiting REPL.")
                break
    else:
        run_single_prompt(prompt)


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
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Enable or disable real-time live token streaming to terminal",
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
    stream,
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
        stream=stream,
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
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Enable or disable real-time live token streaming to terminal",
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
    stream,
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
        stream=stream,
    )
