import os
import argparse
from tiny_llm.data import prepare_and_train_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare dataset and train BPE tokenizer."
    )
    parser.add_argument("input_file", type=str, help="Path to the input TSV file")
    parser.add_argument(
        "--corpus-file",
        type=str,
        default="data/corpus.txt",
        help="Path to output corpus text file",
    )
    parser.add_argument(
        "--tokenizer-out",
        type=str,
        default="checkpoints/tokenizer.json",
        help="Path to output tokenizer.json file",
    )
    parser.add_argument(
        "--scratch-tokenizer",
        action="store_true",
        help="Train using the from-scratch Python tokenizer instead of HuggingFace's Rust library",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.corpus_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.tokenizer_out), exist_ok=True)

    # Example workflow:
    # wget https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
    # bunzip2 ./kor_sentences.tsv.bz2
    # python scripts/prepare_data.py ./kor_sentences.tsv
    prepare_and_train_tokenizer(
        input_file=args.input_file,
        corpus_file=args.corpus_file,
        tokenizer_out=args.tokenizer_out,
        use_scratch_tokenizer=args.scratch_tokenizer,
    )
