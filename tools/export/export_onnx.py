from tiny_llm.export.onnx_exporter import export_onnx


def export_to_onnx(
    model_path=None, tokenizer_path=None, output_path="tiny_llm.onnx", quantize=False
):
    return export_onnx(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        quantize=quantize,
    )


if __name__ == "__main__":
    export_to_onnx()
