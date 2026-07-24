import os
import json
import tempfile
import torch

from tiny_llm import ScratchTokenizer
from scripts.distill_train import train_distilled_student


def test_distillation_student_training_loop():
    """Verify that train_distilled_student trains TinyLLM on synthetic corpus and saves checkpoint."""
    corpus_content = "톰은 메리가 보는 앞에서 책을 읽기 시작했다.\n메리는 웃으며 톰의 이야기를 들어주었다.\n"

    with (
        tempfile.NamedTemporaryFile(
            "w+", suffix=".txt", encoding="utf-8", delete=False
        ) as corpus_f,
        tempfile.NamedTemporaryFile(
            "w+", suffix=".json", encoding="utf-8", delete=False
        ) as tok_f,
        tempfile.NamedTemporaryFile("wb", suffix=".pth", delete=False) as model_f,
    ):
        corpus_f.write(corpus_content)
        corpus_f.flush()

        tokenizer_data = ScratchTokenizer.train(corpus_content, vocab_size=40)
        json.dump(tokenizer_data, tok_f, ensure_ascii=False)
        tok_f.flush()

        train_distilled_student(
            corpus_path=corpus_f.name,
            tokenizer_path=tok_f.name,
            output_path=model_f.name,
            epochs=1,
            batch_size=2,
            lr=1e-3,
            use_moe=False,
        )

        assert os.path.exists(model_f.name), (
            "Distilled student model checkpoint must be created."
        )
        checkpoint = torch.load(model_f.name, map_location="cpu", weights_only=True)
        assert "tok_embeddings.weight" in checkpoint
