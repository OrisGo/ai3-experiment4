"""Deterministic generation and GSM8K answer scoring before or after SFT."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_alignment.local_sft import MODEL_PATH, PREPARED_DIR, read_jsonl, score_completion, write_jsonl
from config_alignment.runtime import compute_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--limit", type=int, default=128, help="0 evaluates the complete split")
    parser.add_argument("--max-input-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 0 or args.max_input_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("limit must be nonnegative and token limits must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment")
    rows = read_jsonl(PREPARED_DIR / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No evaluation examples found; run prepare_gsm8k_sft.py first")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    load_args = {"dtype": compute_dtype(), "device_map": {"": 0}, "local_files_only": True}
    if args.quantized:
        load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_args)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    predictions = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        encoded = tokenizer(
            row["prompt"], return_tensors="pt", truncation=True, max_length=args.max_input_tokens
        ).to(model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        text = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        score = score_completion(text, row["answer"])
        predictions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "generation": text,
                "generated_tokens": int(generated.shape[1] - encoded["input_ids"].shape[1]),
                "seconds": seconds,
                **score,
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"Evaluated {index}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - started
    count = len(predictions)
    summary = {
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "quantized": args.quantized,
        "compute_dtype": str(compute_dtype()),
        "split": args.split,
        "count": count,
        "strict_accuracy": sum(x["strict_correct"] for x in predictions) / count,
        "answer_accuracy": sum(x["answer_correct"] for x in predictions) / count,
        "format_rate": sum(x["format_ok"] for x in predictions) / count,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": sum(x["generated_tokens"] for x in predictions) / elapsed,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
