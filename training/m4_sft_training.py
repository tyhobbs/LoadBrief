# m4_sft_training.py
# Memory-optimized for M4 48GB — avoids swap
# TRL 1.5.1 / Transformers 5.9.0 / PEFT 0.19.1
# python3 m4_sft_training.py

import os
import json
import time
import glob
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ── Configuration ─────────────────────────────────────────────────────
MODEL_NAME    = "meta-llama/Meta-Llama-3-8B-Instruct"
TRAIN_DATA    = "./formatted/sft_train.jsonl"
VAL_DATA      = "./formatted/sft_validation.jsonl"
OUTPUT_DIR    = "./sft_checkpoint"
FINAL_DIR     = "./sft_checkpoint_final"

BATCH_SIZE    = 2        # reduced from 4
GRAD_ACCUM    = 16       # effective batch still 32
LEARNING_RATE = 2e-4
EPOCHS        = 1        # 1 epoch — sufficient for SFT warmup, saves hours
MAX_LENGTH    = 512      # reduced from 1024 — halves activation memory
LORA_RANK     = 32       # reduced from 64 — halves LoRA memory
LORA_ALPHA    = 64

# Cap dataset size to avoid tokenizing 120k examples into RAM at once
# 20k examples is plenty for strong SFT warmup
MAX_TRAIN_EXAMPLES = 20_000
MAX_VAL_EXAMPLES   = 2_000


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


def main():
    print("=" * 55)
    print("  LoadBrief SFT Training — M4 48GB (memory optimized)")
    print("=" * 55)

    assert torch.backends.mps.is_available(), "MPS not available"
    print("\n[1] MPS verified")

    # Load datasets — capped to avoid RAM overflow
    print("\n[2] Loading datasets...")
    if not Path(TRAIN_DATA).exists():
        raise FileNotFoundError(f"{TRAIN_DATA} not found — run format_dataset.py first")

    train_dataset = load_jsonl(TRAIN_DATA, MAX_TRAIN_EXAMPLES)
    val_dataset   = load_jsonl(VAL_DATA,   MAX_VAL_EXAMPLES)
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    print(f"(capped from full dataset to prevent RAM overflow)")

    steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps     = steps_per_epoch * EPOCHS
    est_minutes     = total_steps * 3 / 60  # ~3 sec/step on M4
    print(f"Estimated training time: ~{est_minutes:.0f} minutes")

    # Load tokenizer
    print(f"\n[3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print(f"\n[4] Loading model (float16)...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="mps"
    )
    print(f"Loaded in {time.time()-start:.0f}s — {model.num_parameters()/1e9:.2f}B params")

    # Enable gradient checkpointing to reduce activation memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print("Gradient checkpointing enabled")

    # Apply LoRA
    print(f"\n[5] Applying LoRA (rank={LORA_RANK})...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # SFTConfig
    print(f"\n[6] Configuring training...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_LENGTH,
        dataset_text_field="text",

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,

        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,

        num_train_epochs=EPOCHS,

        fp16=False,
        bf16=False,
        optim="adamw_torch",

        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=10,
        report_to="none",

        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
    )

    # Trainer
    print(f"\n[7] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print(f"\nTraining summary:")
    print(f"  Examples       : {len(train_dataset):,}")
    print(f"  Steps/epoch    : {steps_per_epoch:,}")
    print(f"  Total steps    : {total_steps:,}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Max length     : {MAX_LENGTH}")
    print(f"  LoRA rank      : {LORA_RANK}")
    print(f"  Est. time      : ~{est_minutes:.0f} min")

    # Auto-resume from checkpoint if one exists
    checkpoints = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )
    resume = checkpoints[-1] if checkpoints else None

    if resume:
        print(f"\n[8] Resuming from: {resume}")
    else:
        print(f"\n[8] Starting training at {time.strftime('%H:%M:%S')}...")

    start = time.time()
    trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Save
    print(f"\n[9] Saving to {FINAL_DIR}...")
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    config_out = {
        "model_name":       MODEL_NAME,
        "lora_rank":        LORA_RANK,
        "effective_batch":  BATCH_SIZE * GRAD_ACCUM,
        "learning_rate":    LEARNING_RATE,
        "epochs":           EPOCHS,
        "max_length":       MAX_LENGTH,
        "train_examples":   len(train_dataset),
        "training_minutes": round(elapsed / 60, 1),
        "trl_version":      "1.5.1",
        "device":           "mps_48gb"
    }
    with open(f"{FINAL_DIR}/training_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\nDone. Next: python3 m4_grpo_training.py")


if __name__ == "__main__":
    main()
