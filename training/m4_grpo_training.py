# m4_grpo_training.py
# GRPO RL fine-tuning for TRL 1.5.1 / Transformers 5.9.0 / PEFT 0.19.1
# python3 m4_grpo_training.py

import os
import re
import json
import time
import glob
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ── Configuration ─────────────────────────────────────────────────────
BASE_MODEL         = "meta-llama/Meta-Llama-3-8B-Instruct"
SFT_CHECKPOINT     = "./sft_checkpoint_final"
GRPO_DATA          = "./formatted/grpo_train.jsonl"
OUTPUT_DIR         = "./grpo_checkpoint"
FINAL_DIR          = "./final_model"

BATCH_SIZE         = 2
GRAD_ACCUM         = 8          # effective batch = 16
NUM_GENERATIONS    = 4
MAX_COMPLETION_LEN = 400        # was max_new_tokens in older TRL
LEARNING_RATE      = 5e-6
BETA               = 0.1        # was kl_coeff in older TRL
EPOCHS             = 2

MAX_TRAIN_EXAMPLES = 20_000
MAX_EVAL_EXAMPLES  = 100


def load_jsonl(path: str, max_examples: int = None) -> Dataset:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
                if max_examples and len(examples) >= max_examples:
                    break
    return Dataset.from_list(examples)


def composite_reward(completions, prompts=None, **kwargs) -> list:
    """
    Five-component reward function grounded in sports science.
    Max score: 1.0
    """
    rewards = []
    for completion in completions:
        text  = completion.lower() if isinstance(completion, str) else ""
        score = 0.0

        # 1. Risk level structured declaration (0.30)
        if any(f"risk level: {r}" in text or f"your status: {r}" in text
               for r in ["low", "moderate", "high", "critical"]):
            score += 0.30
        elif any(r in text for r in ["low", "moderate", "high", "critical"]):
            score += 0.15

        # 2. Recommendation specificity (0.25)
        if re.search(r'\d+\s*[–\-]\s*\d+\s*%', text):
            score += 0.25
        elif re.search(r'rpe\s*[<≤]\s*\d', text):
            score += 0.22
        elif re.search(r'\d+\s*(day|week|session)', text):
            score += 0.18
        elif re.search(r'\d+%', text):
            score += 0.15
        elif any(v in text for v in [
            "reduce", "maintain", "rest", "monitor", "suspend"
        ]):
            score += 0.10

        # 3. Overreaching classification (0.20)
        if any(t in text for t in [
            "non-functional overreaching", "functional overreaching",
            "overtraining syndrome", "normal adaptation", "undertraining"
        ]):
            score += 0.20

        # 4. Escalation trigger (0.15)
        if any(t in text for t in [
            "physician", "medical review", "doctor",
            "clinical assessment", "escalate", "seek support"
        ]):
            score += 0.15

        # 5. Appropriate length (0.10)
        wc = len(completion.split()) if isinstance(completion, str) else 0
        if 200 <= wc <= 450:
            score += 0.10
        elif 100 <= wc <= 600:
            score += 0.05

        rewards.append(min(1.0, round(score, 3)))
    return rewards


def main():
    print("=" * 55)
    print("  LoadBrief GRPO Training — M4 48GB")
    print("  TRL 1.5.1 / Transformers 5.9.0 / PEFT 0.19.1")
    print("=" * 55)

    assert torch.backends.mps.is_available(), "MPS not available"
    print("\n[1] MPS verified")

    # Validate reward function
    print("\n[2] Validating reward function...")
    test_scores = composite_reward([
        "Risk Level: HIGH\n\nReduce volume by 30-40% for 4 days. "
        "Non-functional overreaching markers present. RPE ≤ 6. "
        "Physician review if no improvement in 7 days.",
        "The athlete is tired."
    ])
    print(f"  Good brief: {test_scores[0]} (expect >0.7)")
    print(f"  Poor brief: {test_scores[1]} (expect <0.3)")

    # Load datasets
    print(f"\n[3] Loading GRPO dataset...")
    if not Path(GRPO_DATA).exists():
        raise FileNotFoundError(
            f"{GRPO_DATA} not found — run format_dataset.py first"
        )
    train_dataset = load_jsonl(GRPO_DATA, MAX_TRAIN_EXAMPLES)
    all_data      = load_jsonl(GRPO_DATA)
    eval_dataset  = Dataset.from_list(list(all_data)[-MAX_EVAL_EXAMPLES:])
    print(f"Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")

    # Load tokenizer
    print(f"\n[4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load SFT checkpoint
    print(f"\n[5] Loading SFT checkpoint...")
    start = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="mps"
    )
    model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT)
    print(f"Loaded in {time.time()-start:.0f}s")

    # GRPOConfig with correct TRL 1.5.1 argument names
    print(f"\n[6] Configuring GRPO...")
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,

        # Learning
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.03,
        weight_decay=0.01,

        # GRPO specific — correct names for TRL 1.5.1
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LEN,  # was max_new_tokens
        temperature=0.8,
        top_p=0.95,
        beta=BETA,                                 # was kl_coeff

        # Precision
        fp16=False,
        bf16=False,

        # Eval — small set, runs in ~5 min
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=False,

        # MPS requirements
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        # Logging
        logging_steps=5,
        report_to="none",
        log_completions=True,        # prints sample completions — useful
        num_completions_to_print=1,  # just one per log step
    )

    # Trainer
    print(f"\n[7] Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=composite_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"\nGRPO training summary:")
    print(f"  Train examples  : {len(train_dataset):,}")
    print(f"  Steps/epoch     : {steps_per_epoch:,}")
    print(f"  Total steps     : {steps_per_epoch * EPOCHS:,}")
    print(f"  Generations/step: {NUM_GENERATIONS}")
    print(f"  Beta (KL coeff) : {BETA}")
    print(f"  Max completion  : {MAX_COMPLETION_LEN} tokens")

    # Auto-resume from checkpoint if exists
    checkpoints = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )
    resume = checkpoints[-1] if checkpoints else None
    if resume:
        print(f"\n[8] Resuming from: {resume}")
    else:
        print(f"\n[8] Starting GRPO at {time.strftime('%H:%M:%S')}...")

    start = time.time()
    trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - start
    print(f"\nGRPO complete in {elapsed/60:.1f} minutes")

    # Save
    print(f"\n[9] Saving to {FINAL_DIR}...")
    Path(FINAL_DIR).mkdir(exist_ok=True)
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    config_out = {
        "base_model":       BASE_MODEL,
        "sft_checkpoint":   SFT_CHECKPOINT,
        "learning_rate":    LEARNING_RATE,
        "beta":             BETA,
        "num_generations":  NUM_GENERATIONS,
        "epochs":           EPOCHS,
        "training_minutes": round(elapsed / 60, 1),
        "trl_version":      "1.5.1",
        "device":           "mps_48gb"
    }
    with open(f"{FINAL_DIR}/grpo_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\nDone. Next: python3 upload_to_huggingface.py")


if __name__ == "__main__":
    main()
