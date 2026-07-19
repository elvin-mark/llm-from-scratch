import torch
import torch.nn.functional as F
from model import TinyLLM


def generate(
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    num_sentences: int = 5,
    tokenizer_path: str = "tokenizer.json",
    model_path: str = "tiny_llm.pth",
    use_scratch_tokenizer: bool = False,
):
    """
    Generates text using the trained TinyLLM model.

    This function loads the tokenizer and the saved PyTorch model, then performs
    autoregressive token generation using Top-K sampling and Temperature scaling.

    Args:
        max_tokens (int): The maximum number of tokens to generate per sentence.
        temperature (float): Controls randomness. Higher values increase diversity,
                             lower values make the model more deterministic.
        top_k (int): Limits the sampling pool to the top 'k' most probable next tokens.
                     Reduces the chance of generating gibberish.
        num_sentences (int): Number of independent sentences to generate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_scratch_tokenizer:
        from tokenizer import ScratchTokenizer

        tokenizer = ScratchTokenizer.from_file(tokenizer_path)
        print("Using educational Scratch Tokenizer!")
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        print("Using HuggingFace Tokenizer!")

    vocab_size = tokenizer.get_vocab_size()

    # Re-initialize the model with the identical hyper-parameters used during training
    model = TinyLLM(
        vocab_size=vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        ffn_dim=512,
        max_seq_len=64,
    ).to(device)

    try:
        # We explicitly set weights_only=True to resolve future warnings regarding security of pickling.
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(
            f"Model file '{model_path}' not found. Make sure to train the model first."
        )
        return

    # Set model to evaluation mode (disables dropout, affects batchnorm, though neither are used here)
    model.eval()

    # Identify special tokens used for generation control
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    # Initialize the prompt sequence with the starting token
    tokens = [cls_id]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    print("Generating sentences...")
    print("-" * 30)

    for i in range(num_sentences):
        # Start each sentence fresh from the [CLS] token
        current_ids = input_ids.clone()
        generated_token_ids = []

        # Autoregressive loop
        for _ in range(max_tokens):
            with torch.no_grad():
                # Get logits for the current sequence
                logits = model(current_ids)

                # We only care about predicting the *next* token (the last position's output)
                next_token_logits = logits[0, -1, :]

                # 1. Temperature scaling (smooths or sharpens the distribution)
                next_token_logits = next_token_logits / temperature

                # 2. Top-K filtering (remove long-tail probabilities to avoid garbage text)
                if top_k > 0:
                    top_k_threshold = torch.topk(next_token_logits, top_k)[0][
                        ..., -1, None
                    ]
                    indices_to_remove = next_token_logits < top_k_threshold
                    next_token_logits[indices_to_remove] = float("-inf")

                # Convert filtered logits back to probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Sample the next token based on the resulting probability distribution
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Stop early if the model predicts the end-of-sequence token
                if next_token == sep_id:
                    break

                generated_token_ids.append(next_token)

                # Append the new token to our current sequence context and continue
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(
                    device
                )
                current_ids = torch.cat((current_ids, next_token_tensor), dim=1)

                # Hard cap on sequence length based on our model's maximum sequence length
                if current_ids.size(1) >= 64:
                    break

        # Decode the final list of token IDs back into readable strings
        decoded_text = tokenizer.decode(generated_token_ids)
        print(f"Generated {i + 1}: {decoded_text}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scratch-tokenizer",
        action="store_true",
        help="Use the educational from-scratch tokenizer",
    )
    args = parser.parse_args()
    generate(use_scratch_tokenizer=args.scratch_tokenizer)
