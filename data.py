from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import torch
from torch.utils.data import Dataset


def prepare_and_train_tokenizer(
    input_file: str, corpus_file: str, vocab_size: int = 4000, use_scratch_tokenizer: bool = False
):
    """
    Extracts sentences from a TSV dataset and trains a custom BPE tokenizer.

    This script serves two main purposes:
    1. Data Extraction: Reads a TSV file and extracts the sentence column
       into a plain text format suitable for tokenizer training.
    2. Tokenizer Training: Trains a Hugging Face Byte-Pair Encoding (BPE)
       tokenizer on the extracted corpus.

    Args:
        input_file (str): Path to the input TSV file (e.g., 'kor_sentences.tsv').
        corpus_file (str): Path where the extracted raw text will be saved.
        vocab_size (int): Target vocabulary size for the BPE tokenizer. Default is 4000.
    """
    print("Extracting sentences...")
    # Step 1: Read TSV and extract the text column
    # The expected TSV format is: id \t language \t sentence
    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(corpus_file, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                sentence = parts[2]
                f_out.write(sentence + "\n")

    print("Training tokenizer...")
    if use_scratch_tokenizer:
        from tokenizer import ScratchTokenizer
        import json
        
        with open(corpus_file, "r", encoding="utf-8") as f:
            text = f.read()
            
        tokenizer_data = ScratchTokenizer.train(text, vocab_size=vocab_size)
        
        with open("tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer trained with vocab size {len(tokenizer_data['model']['vocab'])} and saved to tokenizer.json")
    else:
        # Step 2: Initialize BPE Tokenizer with an unknown token fallback
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
        # Pre-tokenize by splitting on whitespaces
        tokenizer.pre_tokenizer = Whitespace()
    
        # Define special tokens required for model training and generation
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    
        # Configure the BPE Trainer
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    
        # Train the tokenizer on our extracted corpus text file
        tokenizer.train(files=[corpus_file], trainer=trainer)
    
        # Save the resulting tokenizer configuration
        tokenizer.save("tokenizer.json")
    
        print(
            f"Tokenizer trained with vocab size {tokenizer.get_vocab_size()} and saved to tokenizer.json"
        )


# --- Dataset and Training ---
class SentencesDataset(Dataset):
    """
    Custom Dataset loader for processing raw text sentences into token IDs.
    Applies padding and truncation to enforce a fixed maximum length.
    """

    def __init__(self, file_path, tokenizer_path, max_length=64):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # Enable truncation for tokenization
        from tokenizers.processors import TemplateProcessing

        # We need to process the special tokens manually or with TemplateProcessing
        cls_id = self.tokenizer.token_to_id("[CLS]")
        sep_id = self.tokenizer.token_to_id("[SEP]")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.data = []
        pad_id = self.tokenizer.token_to_id("[PAD]")

        for line in lines:
            encoding = self.tokenizer.encode(line)
            # Add Special Tokens: [CLS] start, [SEP] end
            tokens = [cls_id] + encoding.ids + [sep_id]

            # Truncate or Pad
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [pad_id] * (max_length - len(tokens))
            self.data.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Shift targets by 1: Next token prediction
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input TSV file")
    parser.add_argument("--scratch-tokenizer", action="store_true", help="Train using the from-scratch Python tokenizer instead of HuggingFace's Rust library")
    args = parser.parse_args()

    # wget https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
    # bunzip2 ./kor_sentences.tsv.bz2
    prepare_and_train_tokenizer(input_file=args.input_file, corpus_file="corpus.txt", use_scratch_tokenizer=args.scratch_tokenizer)
