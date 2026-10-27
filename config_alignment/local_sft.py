"""Local GSM8K adaptation of the CS336 Spring 2025 reasoning SFT task."""

from __future__ import annotations

import hashlib
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT.parent / "Qwen2.5-0.5B"
SOURCE_DIR = PROJECT_ROOT / "data" / "gsm8k"
PREPARED_DIR = PROJECT_ROOT / "data" / "gsm8k_sft"
PROMPT_TEMPLATE = (PROJECT_ROOT / "config_alignment" / "prompts" / "r1_zero.prompt").read_text(
    encoding="utf-8"
).rstrip() + "\n"
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def ground_truth(raw_answer: str) -> str:
    if "####" not in raw_answer:
        raise ValueError("GSM8K answer has no #### final-answer marker")
    answer = raw_answer.rsplit("####", 1)[1].strip()
    if numeric_value(answer) is None:
        raise ValueError(f"GSM8K final answer is not numeric: {answer!r}")
    return answer


def numeric_value(text: str) -> Decimal | None:
    matches = NUMBER_PATTERN.findall(text.replace("$", ""))
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def make_example(row: dict, source_index: int) -> dict:
    rationale = row["answer"].rsplit("####", 1)[0]
    rationale = re.sub(r"<<[^>]*>>", "", rationale).strip()
    gold = ground_truth(row["answer"])
    return {
        "id": source_index,
        "question": row["question"],
        "answer": gold,
        "prompt": PROMPT_TEMPLATE.format(question=row["question"]),
        "completion": f"{rationale}\n</think> <answer> {gold} </answer>",
    }


def prepare_data(seed: int = 42, validation_size: int = 128) -> dict:
    train_path = SOURCE_DIR / "train.jsonl"
    test_path = SOURCE_DIR / "test.jsonl"
    train = [make_example(row, i) for i, row in enumerate(read_jsonl(train_path))]
    test = [make_example(row, i) for i, row in enumerate(read_jsonl(test_path))]
    if not 0 < validation_size < len(train):
        raise ValueError("validation_size must be smaller than the training set")
    random.Random(seed).shuffle(train)
    validation, train_pool = train[:validation_size], train[validation_size:]
    write_jsonl(PREPARED_DIR / "train_pool.jsonl", train_pool)
    write_jsonl(PREPARED_DIR / "validation.jsonl", validation)
    write_jsonl(PREPARED_DIR / "test.jsonl", test)
    manifest = {
        "seed": seed,
        "validation_size": len(validation),
        "train_pool_size": len(train_pool),
        "test_size": len(test),
        "prompt_template": "config_alignment/prompts/r1_zero.prompt",
        "train_source_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "test_source_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "note": "GSM8K substitutes for the unavailable CS336 2025 MATH SFT data.",
    }
    (PREPARED_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def score_completion(text: str, gold: str) -> dict:
    format_ok = "</think> <answer>" in text and "</answer>" in text
    answer_sections = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    parsed = numeric_value(answer_sections[-1]) if answer_sections else None
    gold_number = numeric_value(gold)
    return {
        "format_ok": format_ok,
        "parsed_answer": str(parsed) if parsed is not None else None,
        "answer_correct": parsed is not None and parsed == gold_number,
        "strict_correct": format_ok and parsed is not None and parsed == gold_number,
    }
