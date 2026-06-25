# run_baselines.py
# Generate completions from baseline models AND any trained variant.
# Saves outputs to ./baseline_outputs/ for later evaluation.
#
# Built-in baselines:
#   --baseline zero_shot     : Llama 3 8B with no fine-tuning
#   --baseline sft_only      : Your SFT model (no GRPO)
#   --baseline sft_grpo      : Your full SFT + GRPO model (loads ./final_model)
#   --baseline all           : Run all three built-ins
#
# Arbitrary checkpoint (seeds, ablations, LR sweep):
#   --baseline custom --checkpoint ./final_model_seed2_lr5e-06 --output-name seed2_lr5e-06
#   --baseline custom --checkpoint ./final_model_ablation_none --output-name ablation_none
#
# The --checkpoint path is a LoRA adapter directory produced by
# run_grpo_seed.py or run_ablation.py. --output-name sets the filename
# written to ./baseline_outputs/<name>.jsonl (so compile_results.py can
# parse it). If --output-name is omitted, the checkpoint dir basename
# (with a leading "final_model_" stripped) is used.

import os
import json
import time
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

BASE_MODEL     = "meta-llama/Meta-Llama-3-8B-Instruct"
SFT_CHECKPOINT = "./sft_checkpoint_final"
TEST_DATA      = "./formatted/sft_test.jsonl"
OUTPUT_DIR     = "./baseline_outputs"
N_EXAMPLES     = 500   # how many test examples to generate for


def load_test_prompts(path, n=N_EXAMPLES):
    """Load test examples and extract just the prompt portion."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line.strip())
            # Each example has 'text' field with full prompt+completion
            # Split on "### Load Management Brief:" marker
            text = ex.get("text", "")
            if "### Load Management Brief:" in text:
                prompt, _, completion = text.partition("### Load Management Brief:")
                prompts.append({
                    "id": i,
                    "prompt": prompt + "### Load Management Brief:\n",
                    "reference": completion.strip(),
                })
    return prompts


def generate_one(model, tokenizer, prompt, max_new_tokens=400):
    """Generate a single completion."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt from output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in full_text:
        completion = full_text[len(prompt):].strip()
    else:
        # Fallback — just take everything after the input length
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return completion


def resolve_model_config(baseline_name, checkpoint, output_name):
    """
    Decide which adapter to load and what filename to write, given the
    baseline mode. Returns (adapter_path_or_None, output_file_stem).
    Raises ValueError on bad combinations.
    """
    if baseline_name == "zero_shot":
        return None, "zero_shot"

    if baseline_name == "sft_only":
        return SFT_CHECKPOINT, "sft_only"

    if baseline_name == "sft_grpo":
        return "./final_model", "sft_grpo"

    if baseline_name == "custom":
        if not checkpoint:
            raise ValueError(
                "--baseline custom requires --checkpoint <adapter_dir>"
            )
        # Derive a clean output name if none was given:
        # ./final_model_seed2_lr5e-06  ->  seed2_lr5e-06
        if not output_name:
            base = Path(checkpoint.rstrip("/")).name
            output_name = base.replace("final_model_", "", 1)
        return checkpoint, output_name

    raise ValueError(f"Unknown baseline: {baseline_name}")


def run_baseline(baseline_name, prompts, checkpoint=None, output_name=None):
    """Run inference for one model and save results."""
    adapter_path, out_stem = resolve_model_config(
        baseline_name, checkpoint, output_name
    )

    print(f"\n{'='*60}")
    print(f"  Baseline: {baseline_name}"
          + (f"  ({out_stem})" if baseline_name == "custom" else ""))
    print(f"  Output file: {OUTPUT_DIR}/{out_stem}.jsonl")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="mps"
    )

    # Load an adapter on top of the base model unless this is zero_shot
    if adapter_path is not None:
        if not Path(adapter_path).exists():
            print(f"ERROR: adapter path {adapter_path} not found. Skipping.")
            del model
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None
            return
        print(f"Loading adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
    # zero_shot uses base model unchanged

    model.eval()

    # Generate
    results = []
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    output_path = f"{OUTPUT_DIR}/{out_stem}.jsonl"

    # Resume if partial output exists
    start_idx = 0
    if Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        start_idx = len(results)
        print(f"Resuming from example {start_idx}")

    print(f"Generating {len(prompts) - start_idx} completions...")
    t0 = time.time()
    with open(output_path, "a") as f:
        for i, prompt_data in enumerate(prompts[start_idx:], start=start_idx):
            try:
                completion = generate_one(model, tokenizer, prompt_data["prompt"])
            except Exception as e:
                print(f"  Error on {i}: {e}")
                completion = ""

            result = {
                "id":         prompt_data["id"],
                "prompt":     prompt_data["prompt"],
                "completion": completion,
                "reference":  prompt_data["reference"],
                "baseline":   out_stem,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()
            results.append(result)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate    = (i + 1 - start_idx) / elapsed
                eta     = (len(prompts) - i - 1) / max(rate, 0.01)
                print(f"  {i+1}/{len(prompts)} | {rate*60:.1f}/min | ETA {eta/60:.1f} min")

    print(f"\nDone. Saved {len(results)} to {output_path}")
    del model
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True,
                        choices=["zero_shot", "sft_only", "sft_grpo", "custom", "all"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA adapter dir for --baseline custom "
                             "(e.g. ./final_model_seed2_lr5e-06)")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output filename stem under ./baseline_outputs/ "
                             "for --baseline custom. Defaults to the checkpoint "
                             "dir name with 'final_model_' stripped.")
    parser.add_argument("--n", type=int, default=N_EXAMPLES,
                        help="Number of test examples to evaluate (default 500)")
    args = parser.parse_args()

    print(f"Loading {args.n} test prompts...")
    prompts = load_test_prompts(TEST_DATA, args.n)
    print(f"Loaded {len(prompts)} prompts")

    if args.baseline == "all":
        for b in ["zero_shot", "sft_only", "sft_grpo"]:
            run_baseline(b, prompts)
    else:
        run_baseline(
            args.baseline, prompts,
            checkpoint=args.checkpoint,
            output_name=args.output_name,
        )


if __name__ == "__main__":
    main()
