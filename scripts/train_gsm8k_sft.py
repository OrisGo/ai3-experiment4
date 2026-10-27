"""Train and measure GSM8K instruction tuning with LoRA, QLoRA, or full weights."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_alignment.local_sft import MODEL_PATH, PREPARED_DIR, read_jsonl
from config_alignment.runtime import compute_dtype


class WindowsSFTTrainer(SFTTrainer):
    """Use the standard Transformers loss when fused TRL loss is unavailable."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return Trainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def fitting_rows(rows: list[dict], tokenizer, max_length: int) -> tuple[list[dict], int]:
    """Drop examples whose final answer would be lost through truncation."""
    kept = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(
            row["completion"] + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]
        if len(prompt_ids) + len(completion_ids) <= max_length:
            kept.append(row)
    return kept, len(rows) - len(kept)


def make_quantization_config(method: str) -> BitsAndBytesConfig | None:
    """Use 4-bit NF4 and double quantization for QLoRA; otherwise return None."""
    if method != "qlora":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(),
    )

def make_lora_config(method: str, rank: int) -> LoraConfig | None:
    """Set CAUSAL_LM, rank, alpha=2*rank, dropout=0, and all-linear targets."""
    if method == "full":
        return None
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0,
        target_modules="all-linear"
    )

def make_sft_config(args: argparse.Namespace) -> SFTConfig:
    """Configure SFTTrainer for completion-only loss and fair comparisons.

    Use one epoch by default, batch size 1, gradient accumulation 8, max length
    384, evaluation once per epoch, logging progress, gradient checkpointing,
    and a fixed seed. Allow --epochs to override the default. Use learning rate
    2e-4 for LoRA/QLoRA and 2e-5 for full fine-tuning unless overridden; honor
    --optimizer. Use BF16 on the supported CUDA device.
    """
    lr = (
        args.learning_rate
        if args.learning_rate is not None
        else (2e-5 if args.method == "full" else 2e-4)
    )
    return SFTConfig(
        output_dir=str(args.output_dir),
        completion_only_loss=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        eval_strategy="epoch",
        logging_steps=10,
        gradient_checkpointing=True,
        seed=42,
        learning_rate=lr,
        optim=args.optimizer,
        bf16=compute_dtype() == torch.bfloat16,
        fp16=False,
        report_to="none",
    )


def make_trainer(model, tokenizer, train_dataset, eval_dataset, args) -> SFTTrainer:
    """Construct WindowsSFTTrainer with the PEFT and SFT configs."""
    sft_config = make_sft_config(args=args)
    peft_config = make_lora_config(method=args.method, rank=args.rank)

    return WindowsSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config
    )


def count_parameters(model) -> tuple[int, int]:
    """Return (trainable_parameters, total_parameters).

    Use logical parameter counts for quantized weights: a packed Params4bit
    tensor does not have the same numel as its unquantized matrix. PEFT provides
    a parameter-counting API that handles this case.
    """
    if isinstance(model, PeftModel):
        return model.get_nb_trainable_parameters()

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )
    return trainable, total


def train_and_measure(trainer: SFTTrainer) -> tuple[object, float, float, float]:
    """Run training and return result, seconds, allocated GiB, reserved GiB.

    Synchronize CUDA before/after timing. Reset peak memory counters before
    training. Keep the progress bar visible.
    """
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    result = trainer.train()

    if use_cuda:
        torch.cuda.synchronize()

    seconds = time.perf_counter() - start

    if use_cuda:
        allocated_gib = (
            torch.cuda.max_memory_allocated() / 1024**3
        )
        reserved_gib = (
            torch.cuda.max_memory_reserved() / 1024**3
        )
    else:
        allocated_gib = 0.0
        reserved_gib = 0.0
    return result, seconds, allocated_gib, reserved_gib



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--method", choices=["lora", "qlora", "full"], required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=0, help="0 uses all fitting training data")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer", choices=["adamw_torch", "adamw_bnb_8bit"], default="adamw_torch")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment")
    if args.rank < 1 or args.train_size < 0 or args.max_length < 64 or args.epochs <= 0:
        raise ValueError("Invalid rank, training size, sequence length, or epoch count")

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_source = read_jsonl(PREPARED_DIR / "train_pool.jsonl")
    validation_source = read_jsonl(PREPARED_DIR / "validation.jsonl")
    train_rows, train_skipped = fitting_rows(train_source, tokenizer, args.max_length)
    validation_rows, validation_skipped = fitting_rows(validation_source, tokenizer, args.max_length)
    selected = train_rows[: args.train_size] if args.train_size else train_rows
    if not selected or not validation_rows:
        raise ValueError("No complete examples fit max-length; prepare data first")
    if args.train_size and len(selected) != args.train_size:
        raise ValueError(f"Only {len(selected)} complete examples fit max-length={args.max_length}")

    to_dataset = lambda rows: Dataset.from_list(
        [{"prompt": row["prompt"], "completion": row["completion"]} for row in rows]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs = {
        "dtype": compute_dtype(),
        "device_map": {"": 0},
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    quantization_config = make_quantization_config(args.method)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False
    trainer = make_trainer(model, tokenizer, to_dataset(selected), to_dataset(validation_rows), args)

    trainable, total = count_parameters(trainer.model)
    if not 0 < trainable <= total:
        raise ValueError("Invalid trainable/total parameter counts")
    if args.method == "full" and trainable != total:
        raise ValueError("Full fine-tuning must update all model parameters")
    if args.method != "full" and trainable == total:
        raise ValueError("LoRA/QLoRA must freeze the base model parameters")
    print(f"Trainable parameters: {trainable:,} / {total:,}", flush=True)
    result, seconds, peak_allocated_gib, peak_reserved_gib = train_and_measure(trainer)
    trainer.save_model(str(args.output_dir / "model"))
    tokenizer.save_pretrained(str(args.output_dir / "model"))
    final_eval = next((x for x in reversed(trainer.state.log_history) if "eval_loss" in x), {})
    summary = {
        "method": args.method,
        "model_path": str(args.model_path),
        "train_size": len(selected),
        "validation_size": len(validation_rows),
        "train_filtered_too_long": train_skipped,
        "validation_filtered_too_long": validation_skipped,
        "max_length": args.max_length,
        "rank": args.rank if args.method != "full" else None,
        "learning_rate": trainer.args.learning_rate,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "epochs": args.epochs,
        "seed": 42,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "compute_dtype": str(compute_dtype()),
        "optimizer_steps": trainer.state.global_step,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "train_loss": result.training_loss,
        "validation_loss": final_eval.get("eval_loss"),
        "elapsed_seconds": seconds,
        "peak_cuda_allocated_gib": peak_allocated_gib,
        "peak_cuda_reserved_gib": peak_reserved_gib,
        "adapter_or_model_dir": str(args.output_dir / "model"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
