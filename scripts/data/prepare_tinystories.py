"""
Download and prepare the TinyStories dataset (roneneldan/TinyStories)
Extracts story text to data/corpus.txt and trains a subword BPE tokenizer.
"""

import argparse
import json
import os

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from tiny_llm import ScratchTokenizer


def prepare_tinystories(
    output_corpus: str = "data/corpus.txt",
    output_tokenizer: str = "checkpoints/tokenizer.json",
    num_samples: int = 50000,
    vocab_size: int = 4000,
    use_scratch_tokenizer: bool = False,
):
    print("📖 Downloading and Preparing TinyStories Dataset (roneneldan/TinyStories)...")
    os.makedirs(os.path.dirname(os.path.abspath(output_corpus)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_tokenizer)), exist_ok=True)

    # 1. Stream TinyStories dataset using HuggingFace datasets
    print(f"  Fetching {num_samples:,} story samples from Hugging Face...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    lines_count = 0
    with open(output_corpus, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(ds):
            if idx >= num_samples:
                break
            text = sample.get("text", "").strip()
            if text:
                clean_story = " ".join(text.split())
                f.write(clean_story + "\n")
                lines_count += 1

    corpus_size_mb = os.path.getsize(output_corpus) / (1024.0 * 1024.0)
    print(f"✅ Extracted {lines_count:,} stories to '{output_corpus}' ({corpus_size_mb:.2f} MB).")

    # 2. Train Subword BPE Tokenizer
    print(f"\n⚡ Training Subword BPE Tokenizer (vocab_size={vocab_size:,})...")
    if use_scratch_tokenizer:
        with open(output_corpus, "r", encoding="utf-8") as f:
            corpus_text = f.read(500000)
        tok_data = ScratchTokenizer.train(corpus_text, vocab_size=vocab_size)
        with open(output_tokenizer, "w", encoding="utf-8") as f:
            json.dump(tok_data, f, ensure_ascii=False, indent=2)
    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train([output_corpus], trainer)
        tokenizer.save(output_tokenizer)

    print(f"✅ Tokenizer saved to '{output_tokenizer}'. Ready for model training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TinyStories dataset & tokenizer.")
    parser.add_argument("--corpus", type=str, default="data/corpus.txt", help="Output corpus path")
    parser.add_argument(
        "--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Output tokenizer path"
    )
    parser.add_argument(
        "--samples", type=int, default=50000, help="Number of TinyStories samples to download"
    )
    parser.add_argument("--vocab-size", type=int, default=4000, help="Subword BPE vocabulary size")
    parser.add_argument(
        "--scratch-tokenizer", action="store_true", help="Use pure-Python educational BPE trainer"
    )

    args = parser.parse_args()
    prepare_tinystories(
        output_corpus=args.corpus,
        output_tokenizer=args.tokenizer,
        num_samples=args.samples,
        vocab_size=args.vocab_size,
        use_scratch_tokenizer=args.scratch_tokenizer,
    )
