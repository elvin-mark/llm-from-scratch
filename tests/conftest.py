import json
import os

import pytest

from tiny_llm import ScratchTokenizer


@pytest.fixture(scope="session", autouse=True)
def ensure_test_fixtures():
    """
    Session-scoped Pytest fixture that automatically ensures synthetic test fixtures
    (checkpoints/tokenizer.json, data/corpus.txt, checkpoints/vocab.bin) exist
    before running any tests in CI or fresh local environments.
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("c", exist_ok=True)

    # 1. Generate data/corpus.txt if missing
    corpus_path = "data/corpus.txt"
    if not os.path.exists(corpus_path):
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(
                "안녕하세요. 톰은 메리가 보는 앞에서 책을 읽기 시작했다.\n"
                "메리는 웃으며 톰의 이야기를 들어주었다.\n"
                "오늘 날씨가 정말 좋습니다.\n"
            )

    # 2. Generate checkpoints/tokenizer.json if missing
    tokenizer_path = "checkpoints/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        tok_data = ScratchTokenizer.train(
            "안녕하세요. 톰은 메리가 보는 앞에서 책을 읽기 시작했다. 오늘 날씨가 정말 좋습니다.",
            vocab_size=100,
        )
        with open(tokenizer_path, "w", encoding="utf-8") as f:
            json.dump(tok_data, f, ensure_ascii=False, indent=2)

    # 3. Generate checkpoints/vocab.bin if missing
    vocab_bin_path = "checkpoints/vocab.bin"
    if not os.path.exists(vocab_bin_path):
        with open(vocab_bin_path, "wb") as f:
            f.write(b"\0" * 4000 * 32)
