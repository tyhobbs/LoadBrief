# run_ablation.py
# Ablation studies — run GRPO with specific reward components disabled.
# Usage:
#   python3 run_ablation.py --ablation no_signal_conflict
#   python3 run_ablation.py --ablation no_escalation
#   python3 run_ablation.py --ablation no_overreaching

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
SEED               = 42
LR                 = 5e-6


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


def make_reward_function(ablation):
    """Returns a reward function with the specified component disabled."""

    # Redistribute weights when a component is removed
    weights = {
        "risk_level":      0.30,
        "rec_specificity": 0.25,
        "overreaching":    0.20,
        "escalation":      0.15,
        "length":          0.10,
    }

    # Disable the specified component, redistribute proportionally
    if ablation == "no_signal_conflict":
        # Special case — signal conflict is implicit in risk_level reward
        # so we make risk_level only reward simple risk classification
        disabled = "signal_conflict"
    elif ablation in weights:
        weights[ablation] = 0.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    elif ablation == "none":
        pass  # full reward — control baseline
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for completion in completions:
            text  = completion.lower() if isinstance(completion, str) else ""
            score = 0.0

            # 1. Risk level (or simple risk classification for no_signal_conflict)
            if ablation == "no_signal_conflict":
                # Reward only basic risk mention, no structural reward
                if any(r in text for r in ["low", "moderate", "high", "critical"]):
                    score += weights["risk_level"]
            else:
                if any(f"risk level: {r}" in text or f"your status: {r}" in text
                       for r in ["low", "moderate", "high", "critical"]):
                    score += weights["risk_level"]
                elif any(r in text for r in ["low", "moderate", "high", "critical"]):
                    score += weights["risk_level"] * 0.5

            # 2. Recommendation specificity
            if re.search(r'\d+\s*[–\-]\s*\d+\s*%', text):
                score += weights["rec_specificity"]
            elif re.search(r'rpe\s*[<≤]\s*\d', text):
                score += weights["rec_specificity"] * 0.88
            elif re.search(r'\d+\s*(day|week|session)', text):
                score += weights["rec_specificity"] * 0.72
            elif re.search(r'\d+%', text):
                score += weights["rec_specificity"] * 0.60
            elif any(v in text for v in ["reduce", "maintain", "rest", "monitor", "suspend"]):
                score += weights["rec_specificity"] * 0.40

            # 3. Overreaching classification
            if any(t in text for t in [
                "non-functional overreaching", "functional overreaching",
                "overtraining syndrome", "normal adaptation", "undertraining"
            ]):
                score += weights["overreaching"]

            # 4. Escalation trigger
            if any(t in text for t in [
                "physician", "medical review", "doctor",
                "clinical assessment", "escalate", "seek support"
            ]):
                score += weights["escalation"]

            # 5. Length
            wc = len(completion.split()) if isinstance(completion, str) else 0
            if 200 <= wc <= 450:
                score += weights["length"]
            elif 100 <= wc <= 600:
                score += weights["length"] * 0.5

            rewards.append(min(1.0, round(score, 3)))
        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["none", "no_signal_conflict", "no_overreaching",
                                "no_escalation", "no_length"])
    args = parser.parse_args()

    output_dir = f"./grpo_ablation_{args.ablation}"
    final_dir  = f"./final_model_ablation_{args.ablation}"

    print("=" * 60)
    print(f"  Ablation: {args.ablation}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    torch.manual_seed(SEED)
    assert torch.backends.mps.is_available()

    # Data
    train_dataset = load_jsonl(GRPO_DATA, MAX_TRAIN_EXAMPLES)
    print(f"Train: {len(train_dataset)} examples")

    # Model
    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="mps"
    )
    model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT)

    reward_fn = make_reward_function(args.ablation)

    # Validate the reward function
    test_scores = reward_fn([
        "Risk Level: HIGH\n\nReduce volume by 30-40% for 4 days. "
        "Non-functional overreaching markers present. RPE ≤ 6. "
        "Physician review if no improvement in 7 days.",
        "The athlete is tired."
    ])
    print(f"Reward function check: good={test_scores[0]:.2f}, poor={test_scores[1]:.2f}")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        seed=SEED, data_seed=SEED,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.03,
        weight_decay=0.01,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LEN,
        temperature=0.8, top_p=0.95, beta=BETA,
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
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )
    resume = checkpoints[-1] if checkpoints else None

    print(f"\nStarting at {time.strftime('%H:%M:%S')}")
    start = time.time()
    trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - start

    Path(final_dir).mkdir(exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(f"{final_dir}/config.json", "w") as f:
        json.dump({
            "ablation": args.ablation,
            "seed": SEED, "lr": LR,
            "training_minutes": round(elapsed / 60, 1),
        }, f, indent=2)
    print(f"\nSaved to {final_dir}")


if __name__ == "__main__":
    main()
