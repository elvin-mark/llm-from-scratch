from tiny_llm.export.q8_exporter import export_q8


def export_model_q8(model_path=None, tokenizer_path=None, output_path=None):
    return export_q8(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    export_model_q8()
