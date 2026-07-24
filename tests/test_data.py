import os
import torch
from tiny_llm.data import SentencesDataset


def test_dataset_loader(tmp_path):
    """
    Verify the SentencesDataset:
    1. Returns inputs and targets of length max_length - 1.
    2. Target labels are correctly shifted by one index relative to inputs.
    """
    # Create a temporary corpus file
    corpus_content = "안녕하세요\n오늘 날씨가 정말 좋습니다\n"
    corpus_file = tmp_path / "mock_corpus.txt"
    corpus_file.write_text(corpus_content, encoding="utf-8")

    # We use the existing tokenizer.json in the repository
    tokenizer_path = (
        "checkpoints/tokenizer.json"
        if os.path.exists("checkpoints/tokenizer.json")
        else "tokenizer.json"
    )
    assert os.path.exists(tokenizer_path), (
        "tokenizer.json must exist in checkpoints/ or root folder for testing."
    )

    max_length = 16
    dataset = SentencesDataset(
        file_path=str(corpus_file), tokenizer_path=tokenizer_path, max_length=max_length
    )

    # Verify dataset length matches the non-empty lines in mock corpus
    assert len(dataset) == 2

    # Fetch first example
    x, y = dataset[0]

    # Verify tensor shapes
    assert x.shape == (max_length - 1,)
    assert y.shape == (max_length - 1,)

    # Verify next-token prediction shift-by-one labels:
    # y[i] should be the next token after x[i] (which is x[i+1])
    assert torch.equal(y[:-1], x[1:])
