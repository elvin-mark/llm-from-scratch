from tiny_llm.export.c_exporter import export_c


def export_model(model_path=None, tokenizer_path=None, output_path=None, vocab_path=None):
    return export_c(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        vocab_path=vocab_path,
    )


if __name__ == "__main__":
    export_model()
