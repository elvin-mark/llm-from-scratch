# Tokenizer Architecture (Byte-Pair Encoding)

This document explains the from-scratch Python implementation of the Byte-Pair Encoding (BPE) tokenizer used in this repository (`tokenizer.py`). BPE is the same sub-word tokenization algorithm used by GPT, Llama, and most modern Large Language Models.

The goal of sub-word tokenization is to balance vocabulary size and sequence length. Instead of treating every unique word as a token (which creates an endlessly massive vocabulary) or treating every character as a token (which makes sequences too long for the LLM to process), BPE learns to merge frequent character combinations into single tokens.

---

## 1. Pre-Tokenization
Before BPE can even begin—whether during training or inference—the raw text must be pre-tokenized. 

Our implementation mimics HuggingFace's `Whitespace` pre-tokenizer. We use a regular expression `\w+|[^\w\s]` to:
1. Isolate contiguous alphanumeric words.
2. Isolate individual punctuation marks.
3. Completely discard spaces.

For example, `"Hello, world! 톰은메리가"` becomes `["Hello", ",", "world", "!", "톰은메리가"]`. 
This guarantees that BPE will never merge punctuation into a word, preventing messy tokens like `"world!"`.

---

## 2. Training the Tokenizer
The training process (`ScratchTokenizer.train`) analyzes a large text corpus (`corpus.txt`) to figure out which character combinations appear most frequently.

### Step 1: Initialization
We start by breaking every pre-tokenized word into a list of its individual characters. We also inject special control tokens (`[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`) into our base vocabulary.

### Step 2: Frequency Counting
We count the frequencies of every word in the corpus. We then iterate through the words and count how often every **adjacent pair of characters** appears.

### Step 3: Iterative Merging
We loop until we reach the target `vocab_size` (e.g., 4000):
1. **Find the Best Pair**: We identify the most frequently occurring adjacent pair across the entire corpus. For example, if `"e"` and `"l"` appear next to each other 50,000 times, `("e", "l")` becomes our best pair.
2. **Merge**: We append `("e", "l")` to our **`merges`** list. We combine them into a new token `"el"`, assign it an ID, and add it to our **`vocab`**.
3. **Update Corpus**: We iterate through our corpus and replace all adjacent `"e"` and `"l"` characters with the new single token `"el"`.
4. **Repeat**: In the next iteration, the tokenizer might find that `"h"` and `"el"` frequently appear together, merging them into `"hel"`.

### The Result (`tokenizer.json`)
The output of training is two critical mappings:
- **`vocab`**: A dictionary mapping strings to integers (e.g., `{"h": 40, "el": 290, "hel": 405}`).
- **`merges`**: A list of the character pairs in the exact order they were merged. The index in this list dictates the **priority rank** of the merge.

---

## 3. Inference: Encoding
When the user types a prompt into `generate.py`, we must convert that text into a list of integers.

1. **Pre-tokenize**: We run the exact same regex splitting used during training.
2. **Split to Characters**: We break the word into characters (e.g., `"Hello"` -> `['H', 'e', 'l', 'l', 'o']`).
3. **Find Best Merge**: We look at all adjacent pairs in our current list. We consult our `merges` priority list. We find the pair that has the **lowest rank (highest priority)**. 
4. **Merge and Repeat**: We merge that pair in the string list. We repeat this process until no adjacent pairs exist in our `merges` list.
5. **ID Mapping**: We map the final string chunks to their integer IDs using the `vocab`. If a string isn't in the vocab, it defaults to the `[UNK]` ID.

---

## 4. Inference: Decoding
Decoding is the reverse process, where the LLM's integer output is converted back to human text. 
Because we use a simplistic whitespace pre-tokenizer, decoding is as simple as mapping the IDs back to strings via the inverted `vocab` dictionary, and joining them together with a space.
