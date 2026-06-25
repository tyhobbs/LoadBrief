# run_grpo_seed.py
# Wrapper around GRPO training for paper experiments — random seeds and LR sweeps.
# Usage:
#   python3 run_grpo_seed.py --seed 2
#   python3 run_grpo_seed.py --seed 2 --lr 5e-5
#   python3 run_grpo_seed.py --seed 3 --output-suffix seed3

import os
import re
import json
import time
import glob
import argparse
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

BASE_MODEL         = "meta-llama/Meta-Llama-3-8B-Instruct"
SFT_CHECKPOINT     = "./sft_checkpoint_final"
GRPO_DATA          = "./formatted/grpo_train.jsonl"

BATCH_SIZE         = 2
GRAD_ACCUM         = 8
NUM_GENERATIONS    = 2
MAX_COMPLETION_LEN = 400
BETA               = 0.1
EPOCHS             = 1
MAX_TRAIN_EXAMPLES = 3200
MAX_EVAL_EXAMPLES  = 100


def load_jsonl(path, max_examples=None):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
                if max_examples and len(examples) >= max_examples:
                    break
    return Dataset.from_list(examples)


def composite_reward(completions, prompts=None, **kwargs):
    """Standard 5-component reward function."""
    rewards = []
    for completion in completions:
        text  = completion.lower() if isinstance(completion, str) else ""
        score = 0.0
        if any(f"risk level: {r}" in text or f"your status: {r}" in text
               for r in ["low", "moderate", "high", "critical"]):
            score += 0.30
        elif any(r in text for r in ["low", "moderate", "high", "critical"]):
            score += 0.15
        if re.search(r'\d+\s*[–\-]\s*\d+\s*%', text):
            score += 0.25
        elif re.search(r'rpe\s*[<≤]\s*\d', text):
            score += 0.22
        elif re.search(r'\d+\s*(day|week|session)', text):
            score += 0.18
        elif re.search(r'\d+%', text):
            score += 0.15
        elif any(v in text for v in ["reduce", "maintain", "rest", "monitor", "suspend"]):
            score += 0.10
        if any(t in text for t in [
            "non-functional overreaching", "functional overreaching",
            "overtraining syndrome", "normal adaptation", "undertraining"
        ]):
            score += 0.20
        if any(t in text for t in [
            "physician", "medical review", "doctor",
            "clinical assessment", "escalate", "seek support"
        ]):
            score += 0.15
        wc = len(completion.split()) if isinstance(completion, str) else 0
        if 200 <= wc <= 450:
            score += 0.10
        elif 100 <= wc <= 600:
            score += 0.05
        rewards.append(min(1.0, round(score, 3)))
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--lr",            type=float, default=5e-6)
    parser.add_argument("--output-suffix", type=str,   default=None)
    args = parser.parse_args()

    # Auto-generate output suffix if not provided
    suffix = args.output_suffix or f"seed{args.seed}_lr{args.lr:.0e}"
    output_dir = f"./grpo_checkpoint_{suffix}"
    final_dir  = f"./final_model_{suffix}"

    print("=" * 60)
    print(f"  GRPO Run — seed={args.seed}, lr={args.lr}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    torch.manual_seed(args.seed)

    assert torch.backends.mps.is_available()

    # Load data
    train_dataset = load_jsonl(GRPO_DATA, MAX_TRAIN_EXAMPLES)
    print(f"Train: {len(train_dataset)} examples")

    # Load model from SFT checkpoint
    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="mps"
    )
    model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT)

    # Config with this seed and LR
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        seed=args.seed,
        data_seed=args.seed,
        learning_rate=args.lr,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.03,
        weight_decay=0.01,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LEN,
        temperature=0.8,
        top_p=0.95,
        beta=BETA,
        fp16=False, bf16=False,
        eval_strategy="no",
        save_strategy="steps", save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_steps=5,
        report_to="none",
        log_completions=True,
        num_completions_to_print=1,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=composite_reward,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Auto-resume
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )
    resume = checkpoints[-1] if checkpoints else None

    print(f"\nStarting at {time.strftime('%H:%M:%S')}")
    start = time.time()
    trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - start
    print(f"\nComplete in {elapsed/60:.1f} min")

    # Save final
    Path(final_dir).mkdir(exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(f"{final_dir}/config.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "lr": args.lr,
            "epochs": EPOCHS,
            "training_minutes": round(elapsed / 60, 1),
            "max_train_examples": MAX_TRAIN_EXAMPLES,
        }, f, indent=2)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()
