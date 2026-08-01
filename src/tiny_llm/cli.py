import argparse
import os
import sys
import torch
import torch.nn.functional as F

from tiny_llm.data import SentencesDataset, prepare_and_train_tokenizer
from tiny_llm.models.factory import (
    MODEL_REGISTRY,
    create_model,
    load_model_from_checkpoint,
)
from tiny_llm.modules.lora import inject_lora


def handle_train(args):
    """Handler for 'tiny-llm train' command."""
    print(f"🚀 Initializing training for architecture: '{args.arch}'...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Load Tokenizer
    tokenizer_path = args.tokenizer_path
    if not os.path.exists(tokenizer_path):
        # Fallback check
        if os.path.exists("tokenizer.json"):
            tokenizer_path = "tokenizer.json"
        else:
            print(f"Error: Tokenizer file not found at '{tokenizer_path}'.")
            print("Please run 'tiny-llm prepare-data' first.")
            sys.exit(1)

    if args.use_scratch_tokenizer:
        from tiny_llm.tokenizer import ScratchTokenizer

        tokenizer = ScratchTokenizer.from_file(tokenizer_path)
        vocab_size = len(tokenizer.vocab)
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()

    # 2. Check Data
    if not os.path.exists(args.data):
        print(f"Error: Training corpus not found at '{args.data}'.")
        sys.exit(1)

    dataset = SentencesDataset(
        args.data, tokenizer_path, max_length=args.max_seq_len
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    # 3. Create Model & Config
    model, config = create_model(
        arch=args.arch,
        vocab_size=vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.max_seq_len,
        n_kv_heads=args.n_kv_heads,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
    )

    if args.lora:
        print(f"Injecting LoRA adapters (rank={args.lora_rank})...")
        inject_lora(model, r=args.lora_rank, alpha=args.lora_alpha)

    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model initialized: {total_params:,} trainable parameters ({args.arch.upper()})."
    )

    # 4. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(
        f"Starting training loop ({args.epochs} epochs, batch_size={args.batch_size})..."
    )
    for epoch in range(1, args.epochs + 1):
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
        print(f"Epoch [{epoch}/{args.epochs}] - Average Loss: {avg_loss:.4f}")

    # 5. Save Checkpoint & Config JSON
    ckpt_name = f"{args.arch}_model.pth" if args.arch != "dense" else "tiny_llm.pth"
    model_path = os.path.join(args.checkpoint_dir, ckpt_name)
    config_path = os.path.join(args.checkpoint_dir, "config.json")

    torch.save(model.state_dict(), model_path)
    config.save_json(config_path)

    print(f"✅ Training completed! Model saved to '{model_path}'.")
    print(f"✅ Config metadata saved to '{config_path}'.")


def sample_tokens(
    model,
    input_ids,
    max_tokens=64,
    temperature=0.8,
    top_k=50,
    device="cpu",
    max_seq_len=128,
):
    """Autoregressive text token generator helper."""
    model.eval()
    generated = input_ids.clone().to(device)

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


def handle_generate(args):
    """Handler for 'tiny-llm generate' / 'tiny-llm infer' command."""
    checkpoint_path = args.checkpoint
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
            print(f"Error: Model checkpoint not found at '{checkpoint_path}'.")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from '{checkpoint_path}' on {device}...")

    # Auto-detect architecture & config from checkpoint
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    print(
        f"Detected architecture: '{config.arch.upper()}' (dim={config.dim}, layers={config.n_layers}, heads={config.n_heads})."
    )

    # Load Tokenizer
    tokenizer_path = args.tokenizer_path
    if not os.path.exists(tokenizer_path):
        if os.path.exists("checkpoints/tokenizer.json"):
            tokenizer_path = "checkpoints/tokenizer.json"
        elif os.path.exists("tokenizer.json"):
            tokenizer_path = "tokenizer.json"

    if args.use_scratch_tokenizer:
        from tiny_llm.tokenizer import ScratchTokenizer

        tokenizer = ScratchTokenizer.from_file(tokenizer_path)
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)

    # Encode Helper
    def encode_text(prompt):
        if args.use_scratch_tokenizer:
            cls_id = tokenizer.vocab.get("[CLS]", 1)
            return [cls_id] + tokenizer.encode(prompt)
        else:
            cls_id = tokenizer.token_to_id("[CLS]")
            ids = tokenizer.encode(prompt).ids
            return [cls_id] + ids if cls_id is not None else ids

    # Decode Helper
    def decode_ids(ids):
        if args.use_scratch_tokenizer:
            return tokenizer.decode(ids)
        else:
            return tokenizer.decode(ids)

    # Interactive REPL mode
    if args.interactive:
        print("\n💬 Entering interactive REPL chat mode (press Ctrl+C to exit):\n")
        while True:
            try:
                user_prompt = input("Prompt > ").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ("exit", "quit"):
                    break

                prompt_ids = torch.tensor(
                    [encode_text(user_prompt)], dtype=torch.long
                )
                output_ids = sample_tokens(
                    model,
                    prompt_ids,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device,
                    max_seq_len=config.max_seq_len,
                )
                text = decode_ids(output_ids)
                print(f"Response: {text}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting REPL.")
                break
    else:
        prompt = args.prompt
        print(f"Prompt: '{prompt}'")
        prompt_ids = torch.tensor([encode_text(prompt)], dtype=torch.long)

        output_ids = sample_tokens(
            model,
            prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            max_seq_len=config.max_seq_len,
        )
        output_text = decode_ids(output_ids)
        print(f"\nGenerated Output:\n{output_text}")


def handle_prepare_data(args):
    """Handler for 'tiny-llm prepare-data' command."""
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    print(f"Processing input file '{args.input}'...")
    prepare_and_train_tokenizer(
        input_file=args.input,
        corpus_file=args.corpus_file,
        vocab_size=args.vocab_size,
        use_scratch_tokenizer=args.scratch_tokenizer,
        tokenizer_out=args.tokenizer_out,
    )
    print("✅ Data preparation and tokenizer training completed.")


def handle_export(args):
    """Handler for 'tiny-llm export' command."""
    fmt = args.format.lower()
    print(f"Exporting checkpoint '{args.checkpoint}' to format '{fmt}'...")

    if fmt == "onnx":
        from tools.export.export_onnx import export_onnx

        export_onnx(model_path=args.checkpoint, output_dir=args.output_dir)
    elif fmt == "c":
        from tools.export.export_c import export_c

        export_c(model_path=args.checkpoint, output_path=os.path.join(args.output_dir, "model.bin"))
    elif fmt == "q8":
        from tools.export.export_q8 import export_q8

        export_q8(model_path=args.checkpoint, output_path=os.path.join(args.output_dir, "model_q8.bin"))
    elif fmt == "bitnet":
        from tools.export.export_bitnet import export_bitnet

        export_bitnet(model_path=args.checkpoint, output_path=os.path.join(args.output_dir, "model_bitnet.bin"))
    else:
        print(f"Error: Unsupported format '{fmt}'. Supported formats: onnx, c, q8, bitnet.")
        sys.exit(1)


def handle_bench(args):
    """Handler for 'tiny-llm bench' command."""
    suite = args.suite.lower()
    print(f"Running benchmark suite: '{suite}'...")

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
        print(f"Error: Unknown benchmark suite '{suite}'.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="tiny-llm",
        description="Unified CLI Tool for TinyLLM: Train, Infer, Prepare Data, Export, and Benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # ----------------------------------------------------
    # Subcommand: train
    # ----------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train a custom LLM model")
    train_parser.add_argument(
        "--arch",
        choices=list(MODEL_REGISTRY.keys()),
        default="dense",
        help="Model architecture (dense, moe, nano, bitnet). Default: dense.",
    )
    train_parser.add_argument(
        "--data",
        default="data/corpus.txt",
        help="Path to corpus text file. Default: data/corpus.txt",
    )
    train_parser.add_argument(
        "--tokenizer-path",
        default="checkpoints/tokenizer.json",
        help="Path to trained tokenizer.json. Default: checkpoints/tokenizer.json",
    )
    train_parser.add_argument(
        "--use-scratch-tokenizer",
        action="store_true",
        help="Use educational ScratchTokenizer instead of HuggingFace Tokenizer",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    train_parser.add_argument(
        "--dim", type=int, default=128, help="Embedding dimension size"
    )
    train_parser.add_argument(
        "--n-layers", type=int, default=4, help="Number of transformer layers"
    )
    train_parser.add_argument(
        "--n-heads", type=int, default=4, help="Number of attention heads"
    )
    train_parser.add_argument(
        "--ffn-dim", type=int, default=512, help="Feed-forward network hidden dimension"
    )
    train_parser.add_argument(
        "--max-seq-len", type=int, default=64, help="Maximum sequence length"
    )
    train_parser.add_argument(
        "--n-kv-heads", type=int, default=2, help="KV heads count for GQA (MoE model)"
    )
    train_parser.add_argument(
        "--num-experts", type=int, default=8, help="Total experts count (MoE model)"
    )
    train_parser.add_argument(
        "--num-experts-per-tok",
        type=int,
        default=2,
        help="Top-K experts per token (MoE model)",
    )
    train_parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory to save model checkpoints"
    )
    train_parser.add_argument(
        "--lora", action="store_true", help="Inject LoRA adapters for parameter-efficient tuning"
    )
    train_parser.add_argument(
        "--lora-rank", type=int, default=8, help="LoRA rank dimension (if --lora is set)"
    )
    train_parser.add_argument(
        "--lora-alpha", type=int, default=16, help="LoRA scaling alpha (if --lora is set)"
    )

    # ----------------------------------------------------
    # Subcommand: generate / infer
    # ----------------------------------------------------
    gen_parser = subparsers.add_parser(
        "generate", aliases=["infer"], help="Generate text / run inference on a model"
    )
    gen_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to trained model checkpoint (.pth)",
    )
    gen_parser.add_argument(
        "--tokenizer-path",
        default="checkpoints/tokenizer.json",
        help="Path to tokenizer configuration JSON",
    )
    gen_parser.add_argument(
        "--use-scratch-tokenizer",
        action="store_true",
        help="Use educational ScratchTokenizer",
    )
    gen_parser.add_argument(
        "--prompt", default="Once upon a time", help="Text prompt to initialize generation"
    )
    gen_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive REPL mode in terminal",
    )
    gen_parser.add_argument(
        "--max-tokens", type=int, default=64, help="Maximum number of tokens to generate"
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    gen_parser.add_argument(
        "--top-k", type=int, default=50, help="Top-K sampling limit"
    )

    # ----------------------------------------------------
    # Subcommand: prepare-data
    # ----------------------------------------------------
    data_parser = subparsers.add_parser(
        "prepare-data", help="Process dataset and train BPE tokenizer"
    )
    data_parser.add_argument(
        "--input", required=True, help="Path to input text or TSV file"
    )
    data_parser.add_argument(
        "--corpus-file", default="data/corpus.txt", help="Path to output corpus file"
    )
    data_parser.add_argument(
        "--tokenizer-out",
        default="checkpoints/tokenizer.json",
        help="Path to output tokenizer JSON",
    )
    data_parser.add_argument(
        "--vocab-size", type=int, default=4000, help="Target BPE vocabulary size"
    )
    data_parser.add_argument(
        "--scratch-tokenizer",
        action="store_true",
        help="Train using from-scratch Python tokenizer",
    )

    # ----------------------------------------------------
    # Subcommand: export
    # ----------------------------------------------------
    export_parser = subparsers.add_parser(
        "export", help="Export model checkpoint to ONNX, C binary, or Quantized formats"
    )
    export_parser.add_argument(
        "--checkpoint", default="checkpoints/tiny_llm.pth", help="Input PyTorch checkpoint path"
    )
    export_parser.add_argument(
        "--format",
        choices=["onnx", "c", "q8", "bitnet"],
        default="onnx",
        help="Export target format (onnx, c, q8, bitnet)",
    )
    export_parser.add_argument(
        "--output-dir", default="ui/assets", help="Target output directory for exported assets"
    )

    # ----------------------------------------------------
    # Subcommand: bench
    # ----------------------------------------------------
    bench_parser = subparsers.add_parser(
        "bench", help="Run benchmark suite on model modules"
    )
    bench_parser.add_argument(
        "--suite",
        choices=["attention", "flash_attn", "moe", "kv_cache", "bitnet"],
        default="attention",
        help="Benchmark suite to run",
    )

    args = parser.parse_args()

    if args.command == "train":
        handle_train(args)
    elif args.command in ("generate", "infer"):
        handle_generate(args)
    elif args.command == "prepare-data":
        handle_prepare_data(args)
    elif args.command == "export":
        handle_export(args)
    elif args.command == "bench":
        handle_bench(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
