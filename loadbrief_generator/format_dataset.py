# format_dataset.py
# Formats the LoadBrief dataset for multi-audience training.
# Creates 3 training examples per monitoring narrative
# (one per audience type) tripling effective training data.
# python3 format_dataset.py

from datasets import load_dataset, Dataset
import json
from pathlib import Path


def format_multi_audience(example: dict) -> list:
    """
    Create 3 training examples per monitoring narrative.
    One for each audience: athlete, coach, sports_scientist.
    """
    examples = []

    audience_outputs = {
        "athlete":          example.get("output_athlete", ""),
        "coach":            example.get("output_coach", ""),
        "sports_scientist": example.get("output_sports_scientist", "")
    }

    for audience, output in audience_outputs.items():
        # Skip if output is missing or too short
        if not output or len(output.strip()) < 50:
            continue

        examples.append({
            "text": (
                f"### Monitoring Narrative:\n"
                f"{example['input_narrative']}\n\n"
                f"### Audience: {audience}\n\n"
                f"### Load Management Brief:\n"
                f"{output}"
            ),
            "input_narrative": example["input_narrative"],
            "audience": audience,
            "output": output,
            "risk_level": example.get("risk_level", ""),
            "overreaching_classification": example.get(
                "overreaching_classification", ""
            ),
            "complexity_tier": example.get("complexity_tier", 1),
            "sport": example.get("sport", ""),
            "scenario_type": example.get("scenario_type", ""),
            "quality_score": example.get("quality_score", 0.0)
        })

    return examples


def format_prompt_only(example: dict) -> dict:
    """
    Format prompt-only version for GRPO training.
    GRPO generates the completion itself — only needs the prompt.
    """
    return {
        "prompt": (
            f"### Monitoring Narrative:\n"
            f"{example['input_narrative']}\n\n"
            f"### Audience: {example.get('audience', 'coach')}\n\n"
            f"### Load Management Brief:\n"
        ),
        "risk_level": example.get("risk_level", ""),
        "overreaching_classification": example.get(
            "overreaching_classification", ""
        ),
        "complexity_tier": example.get("complexity_tier", 1)
    }


def load_from_jsonl(path: str) -> list:
    """Load examples from JSONL file"""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main():
    print("LoadBrief Dataset Formatter")
    print("=" * 40)

    # Check if dataset exists locally or load from HuggingFace
    local_train = Path("./dataset/train.jsonl")

    if local_train.exists():
        print("Loading from local dataset...")
        train_raw = load_from_jsonl("./dataset/train.jsonl")
        val_raw = load_from_jsonl("./dataset/validation.jsonl")
        test_raw = load_from_jsonl("./dataset/test.jsonl")
    else:
        print("Loading from HuggingFace...")
        print("(Update 'your-username' with your HF username)")
        dataset = load_dataset("your-username/LoadBrief-50K")
        train_raw = list(dataset["train"])
        val_raw = list(dataset["validation"])
        test_raw = list(dataset["test"])

    print(f"Raw examples — Train: {len(train_raw):,} "
          f"| Val: {len(val_raw):,} "
          f"| Test: {len(test_raw):,}")

    # ── SFT Format: Multi-audience expansion ─────────────────────────
    print("\nFormatting for SFT training (multi-audience)...")

    sft_train = []
    for ex in train_raw:
        sft_train.extend(format_multi_audience(ex))

    sft_val = []
    for ex in val_raw:
        sft_val.extend(format_multi_audience(ex))

    sft_test = []
    for ex in test_raw:
        sft_test.extend(format_multi_audience(ex))

    print(f"SFT examples — Train: {len(sft_train):,} "
          f"| Val: {len(sft_val):,} "
          f"| Test: {len(sft_test):,}")
    print(f"Expansion ratio: {len(sft_train)/len(train_raw):.1f}x "
          f"(one per audience type)")

    # Save SFT formatted data
    Path("./formatted").mkdir(exist_ok=True)

    for split_name, split_data in [
        ("train", sft_train),
        ("validation", sft_val),
        ("test", sft_test)
    ]:
        output_path = f"./formatted/sft_{split_name}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Saved {output_path} ({len(split_data):,} examples)")

    # ── GRPO Format: Prompt-only ──────────────────────────────────────
    print("\nFormatting for GRPO training (prompt-only, coach audience)...")

    # GRPO uses coach audience — highest value for sports practitioners
    grpo_train = [
        format_prompt_only({**ex, "audience": "coach"})
        for ex in train_raw
    ]

    output_path = "./formatted/grpo_train.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in grpo_train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved {output_path} ({len(grpo_train):,} examples)")

    # ── Dataset statistics ────────────────────────────────────────────
    print("\nDataset statistics:")

    # Audience distribution in SFT
    audience_counts = {}
    for ex in sft_train:
        aud = ex.get("audience", "unknown")
        audience_counts[aud] = audience_counts.get(aud, 0) + 1
    for aud, count in sorted(audience_counts.items()):
        print(f"  {aud}: {count:,}")

    # Complexity distribution
    tier_counts = {}
    for ex in train_raw:
        t = ex.get("complexity_tier", 0)
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print("\nComplexity tiers (raw):")
    for tier, count in sorted(tier_counts.items()):
        label = {1: 'Clear-cut', 2: 'Moderate', 3: 'Complex'}.get(
            tier, 'Unknown'
        )
        print(f"  Tier {tier} ({label}): {count:,} "
              f"({count/len(train_raw):.1%})")

    # Risk distribution
    risk_counts = {}
    for ex in train_raw:
        risk = ex.get("risk_level", "unknown")
        # Normalize compound risk levels
        risk_key = risk.split("_")[0] if "_" in risk else risk
        risk_counts[risk_key] = risk_counts.get(risk_key, 0) + 1
    print("\nRisk levels (raw train):")
    for risk, count in sorted(risk_counts.items()):
        print(f"  {risk}: {count:,} ({count/len(train_raw):.1%})")

    print("\nFormatting complete.")
    print("Next step: python3 m4_sft_training.py")


if __name__ == "__main__":
    main()
