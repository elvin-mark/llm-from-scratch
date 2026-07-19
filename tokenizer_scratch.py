import json
import re

class ScratchTokenizer:
    """
    A from-scratch implementation of the BPE tokenizer used in TinyLLM.
    It reads 'tokenizer.json' to ensure it produces the exact same tokens
    as the HuggingFace tokenizers library.
    """
    def __init__(self, tokenizer_file="tokenizer.json"):
        with open(tokenizer_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['model']['vocab']
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_id = self.vocab.get(data['model'].get('unk_token', '[UNK]'), 0)
        
        # In tokenizer.json, merges are stored as a list of strings like "A B"
        # The index in this list is the merge priority (lower index = higher priority)
        merges = data['model']['merges']
        self.bpe_ranks = {}
        for i, parts in enumerate(merges):
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = i

    @classmethod
    def from_file(cls, tokenizer_file="tokenizer.json"):
        return cls(tokenizer_file)
                
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
            min_rank = float('inf')
            best_pair = None
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair
                    
            if best_pair is None:
                break # No more merges possible
                
            # Perform merge
            new_word_list = []
            i = 0
            while i < len(word_list):
                if i < len(word_list) - 1 and word_list[i] == best_pair[0] and word_list[i+1] == best_pair[1]:
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
        words = re.findall(r'\w+|[^\w\s]', text)
        
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
