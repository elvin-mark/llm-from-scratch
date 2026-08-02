from tiny_llm.export.bitnet_exporter import export_bitnet as _export_bitnet


def export_bitnet(model_path: str = None, tokenizer_path: str = None, output_path: str = None):
    return _export_bitnet(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    export_bitnet()
