import json
import re


class ScratchTokenizer:
    """
    A from-scratch implementation of the BPE tokenizer used in TinyLLM.
    It reads 'tokenizer.json' to ensure it produces the exact same tokens
    as the HuggingFace tokenizers library.
    """

    def __init__(self, tokenizer_file="tokenizer.json"):
        with open(tokenizer_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["model"]["vocab"]
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_id = self.vocab.get(data["model"].get("unk_token", "[UNK]"), 0)

        # In tokenizer.json, merges are stored as a list of strings like "A B"
        # The index in this list is the merge priority (lower index = higher priority)
        merges = data["model"]["merges"]
        self.bpe_ranks = {}
        for i, parts in enumerate(merges):
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = i

    @classmethod
    def from_file(cls, tokenizer_file="tokenizer.json"):
        return cls(tokenizer_file)

    @classmethod
    def train(cls, text: str, vocab_size: int = 4000, special_tokens: list = None):
        """
        Trains a BPE tokenizer from scratch given a raw text corpus.
        Returns a dictionary perfectly formatted to HuggingFace's tokenizer.json spec.
        """
        if special_tokens is None:
            special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

        print("Pre-tokenizing corpus...")
        words = re.findall(r"\w+|[^\w\s]", text)

        from collections import defaultdict

        word_freqs = defaultdict(int)
        for w in words:
            word_freqs[tuple(w)] += 1

        # Initialize base vocab with all unique characters
        vocab = {}
        for token in special_tokens:
            vocab[token] = len(vocab)

        for word in word_freqs.keys():
            for char in word:
                if char not in vocab:
                    vocab[char] = len(vocab)

        merges = []

        print(f"Base vocabulary size: {len(vocab)}. Target: {vocab_size}")

        while len(vocab) < vocab_size:
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair_freqs[(word[i], word[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Add to merges and vocab
            merges.append(list(best_pair))
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)

            # Merge the best pair in all words
            new_word_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    new_word_freqs[word] = freq
                    continue

                new_word = []
                i = 0
                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == best_pair[0]
                        and word[i + 1] == best_pair[1]
                    ):
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

            if len(vocab) % 500 == 0:
                print(f"Vocab size: {len(vocab)}/{vocab_size}")

        # Format as HF tokenizer.json
        data = {
            "version": "1.0",
            "added_tokens": [
                {
                    "id": i,
                    "content": token,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
                for i, token in enumerate(special_tokens)
            ],
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {
                "type": "BPE",
                "unk_token": "[UNK]",
                "vocab": vocab,
                "merges": merges,
            },
        }
        return data

    def get_vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def token_to_id(self, token):
        return self.vocab.get(token)

    def _get_pairs(self, word_list):
        pairs = set()
        if len(word_list) < 2:
            return pairs
        prev_char = word_list[0]
        for char in word_list[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, word):
        word_list = list(word)
        if len(word_list) < 2:
            return word_list

        while True:
            pairs = self._get_pairs(word_list)
            if not pairs:
                break

            # Find the pair with the lowest rank
            min_rank = float("inf")
            best_pair = None
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if best_pair is None:
                break  # No more merges possible

            # Perform merge
            new_word_list = []
            i = 0
            while i < len(word_list):
                if (
                    i < len(word_list) - 1
                    and word_list[i] == best_pair[0]
                    and word_list[i + 1] == best_pair[1]
                ):
                    new_word_list.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word_list.append(word_list[i])
                    i += 1
            word_list = new_word_list

        return word_list

    class _Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text):
        # 1. Pre-tokenize (Mimics HuggingFace Whitespace pre-tokenizer)
        # It splits by whitespace and punctuation, keeping words and punctuation separate, and drops spaces.
        words = re.findall(r"\w+|[^\w\s]", text)

        # 2. Apply BPE merging on each word
        tokens = []
        for word in words:
            bpe_tokens = self._bpe(word)
            for t in bpe_tokens:
                tokens.append(self.vocab.get(t, self.unk_id))

        # Return an object that has an .ids attribute to match HF interface
        return self._Encoding(tokens)

    def decode(self, ids):
        # By default, HF tokenizer concatenates and joins with space if no decoder is provided.
        return " ".join([self.inv_vocab.get(i, "") for i in ids])


# Example usage/test
if __name__ == "__main__":
    from tokenizers import Tokenizer

    text = "Hello, world! 톰은메리가"

    print("--- HuggingFace Tokenizer ---")
    hf_tokenizer = Tokenizer.from_file("tokenizer.json")
    hf_encoded = hf_tokenizer.encode(text).ids
    print(f"Tokens: {hf_encoded}")
    print(f"Decoded: {hf_tokenizer.decode(hf_encoded)}")

    print("\\n--- Scratch Tokenizer ---")
    scratch_tokenizer = ScratchTokenizer.from_file("tokenizer.json")
    scratch_encoded = scratch_tokenizer.encode(text).ids
    print(f"Tokens: {scratch_encoded}")
    print(f"Decoded: {scratch_tokenizer.decode(scratch_encoded)}")

    assert hf_encoded == scratch_encoded, "Mismatch between tokenizers!"
    print("\\n✅ Perfect Match!")
