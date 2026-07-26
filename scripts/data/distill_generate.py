import os
import time
import argparse
import urllib.request
import json


def query_llm_teacher(
    endpoint_url: str, model_name: str, prompt: str, api_key: str = None
) -> str:
    """
    Queries a Teacher LLM using an OpenAI-compatible API endpoint (works with Ollama, DeepSeek, or vLLM).
    """
    url = f"{endpoint_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant for Korean language dataset generation. Generate natural, fluent Korean text without markdown explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ API call failed: {e}")
        return None


def generate_distilled_dataset(
    input_corpus: str,
    output_corpus: str,
    endpoint_url: str,
    model_name: str,
    max_samples: int = 100,
    api_key: str = None,
):
    print("🚀 Starting Sequence-Level Knowledge Distillation Generation...")
    print(f"  Teacher Model: {model_name}")
    print(f"  Endpoint:      {endpoint_url}")

    if not os.path.exists(input_corpus):
        print(f"❌ Input corpus file '{input_corpus}' not found.")
        return

    with open(input_corpus, "r", encoding="utf-8") as f:
        seed_sentences = [line.strip() for line in f if line.strip()][:max_samples]

    os.makedirs(os.path.dirname(output_corpus), exist_ok=True)
    generated_count = 0

    prompt_templates = [
        "다음 단문 한국어 문장을 풍부하고 자연스러운 복문으로 확장하세요 (문장만 출력):\n",
        "다음 문장을 바탕으로 자연스러운 2턴 한국어 대화를 작성하세요 (대화 내용만 출력):\n",
        "다음 한국어 문장을 다른 자연스러운 표현으로 의역하세요 (문장만 출력):\n",
    ]

    with open(output_corpus, "w", encoding="utf-8") as f_out:
        for idx, sentence in enumerate(seed_sentences, 1):
            print(f"[{idx}/{len(seed_sentences)}] Processing: '{sentence}'")

            for template in prompt_templates:
                prompt = template + f'"{sentence}"'
                response = query_llm_teacher(
                    endpoint_url, model_name, prompt, api_key=api_key
                )

                if response:
                    lines = [
                        line_str.strip()
                        for line_str in response.split("\n")
                        if line_str.strip()
                    ]
                    for line in lines:
                        # Clean unwanted quotes/markdown formatting
                        clean_line = line.replace('"', "").replace("`", "").strip()
                        if len(clean_line) > 3:
                            f_out.write(clean_line + "\n")
                            generated_count += 1

            f_out.flush()
            time.sleep(0.2)

    print(
        f"✅ Distillation complete! Generated {generated_count} synthetic sentences in '{output_corpus}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence-Level Knowledge Distillation Generator."
    )
    parser.add_argument(
        "--input-corpus",
        type=str,
        default="data/corpus.txt",
        help="Seed dataset text file",
    )
    parser.add_argument(
        "--output-corpus",
        type=str,
        default="data/distilled_corpus.txt",
        help="Output distilled text file",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:11434/v1",
        help="Ollama / OpenAI / DeepSeek endpoint URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Teacher model name (e.g. qwen2.5:7b, deepseek-r1:7b)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of seed sentences to process",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for remote providers",
    )

    args = parser.parse_args()

    generate_distilled_dataset(
        input_corpus=args.input_corpus,
        output_corpus=args.output_corpus,
        endpoint_url=args.endpoint,
        model_name=args.model,
        max_samples=args.max_samples,
        api_key=args.api_key,
    )
