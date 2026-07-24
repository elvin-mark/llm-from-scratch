import json
import tempfile
from tiny_llm import ScratchTokenizer


def test_scratch_tokenizer_training_and_serialization():
    """Verify from-scratch BPE training, JSON formatting, and loading."""
    corpus = "안녕하세요. 톰은 메리를 좋아합니다. 메리도 톰을 좋아합니다."

    # Train BPE tokenizer with target vocab size 50
    tokenizer_data = ScratchTokenizer.train(corpus, vocab_size=50)

    assert "model" in tokenizer_data
    assert "vocab" in tokenizer_data["model"]
    assert "merges" in tokenizer_data["model"]
    assert len(tokenizer_data["model"]["vocab"]) <= 50

    # Save to temp file and load via ScratchTokenizer.from_file
    with tempfile.NamedTemporaryFile(
        "w+", suffix=".json", encoding="utf-8", delete=False
    ) as f:
        json.dump(tokenizer_data, f, ensure_ascii=False)
        temp_path = f.name

    tokenizer = ScratchTokenizer.from_file(temp_path)

    assert tokenizer.get_vocab_size() == len(tokenizer_data["model"]["vocab"])
    assert tokenizer.token_to_id("[CLS]") is not None
    assert tokenizer.token_to_id("[SEP]") is not None


def test_scratch_tokenizer_encode_decode():
    """Verify encode and decode behavior for trained text."""
    corpus = "톰은 메리가 우는 것을 보았다."
    tokenizer_data = ScratchTokenizer.train(corpus, vocab_size=60)

    with tempfile.NamedTemporaryFile(
        "w+", suffix=".json", encoding="utf-8", delete=False
    ) as f:
        json.dump(tokenizer_data, f, ensure_ascii=False)
        temp_path = f.name

    tokenizer = ScratchTokenizer.from_file(temp_path)

    encoded = tokenizer.encode("톰은 메리가").ids
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)
