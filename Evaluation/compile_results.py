# compile_results.py
# Compile all experiment results into paper-ready tables and plots.
# Reads from:
#   ./evaluation_results.json     (from evaluate_all.py)
#   ./llm_judge_results/summary.json (from llm_judge.py)
#   ./final_model_seed*/config.json (training metadata)
#   ./final_model_ablation_*/config.json
#
# Outputs:
#   ./paper_artifacts/main_results_table.md   — primary results
#   ./paper_artifacts/ablation_table.md        — ablation analysis
#   ./paper_artifacts/seed_variance.md         — confidence intervals
#   ./paper_artifacts/reward_components.json   — full breakdown for figures
#   ./paper_artifacts/training_curves.png      — loss/reward curves (if matplotlib)

import os
import re
import json
import glob
import statistics
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = "./paper_artifacts"


def load_json(path, default=None):
    if not Path(path).exists():
        return default if default is not None else {}
    with open(path, "r") as f:
        return json.load(f)


def write_md_table(rows, headers, path, title=""):
    """Write a markdown table to file."""
    lines = []
    if title:
        lines.append(f"# {title}\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main_results_table(eval_results, judge_results):
    """Primary comparison table — model performance across metrics."""
    headers = ["Model", "Reward ↑", "Risk Acc ↑", "Over Acc ↑", "Words", "LLM Score ↑"]

    # Order baselines from worst to best
    order = ["zero_shot", "sft_only", "sft_grpo"]
    rows = []
    for name in order:
        if name not in eval_results:
            continue
        e = eval_results[name]
        j = judge_results.get(name, {})

        # Pretty name
        pretty = {
            "zero_shot": "Llama 3 8B (zero-shot)",
            "sft_only":  "+ SFT",
            "sft_grpo":  "+ SFT + GRPO (ours)"
        }.get(name, name)

        rows.append([
            pretty,
            f"{e['reward_mean']:.3f} ± {e['reward_std']:.3f}",
            f"{e['risk_accuracy']:.3f}",
            f"{e['overreaching_accuracy']:.3f}",
            int(e["word_count_mean"]),
            f"{j.get('overall_mean', '—')}" if j else "—",
        ])

    return rows, headers


def ablation_table(eval_results):
    """Ablation study — show what happens when each reward component is removed."""
    headers = ["Ablation", "Reward ↑", "Risk Acc", "Over Acc", "Δ from full"]

    full = eval_results.get("ablation_none")
    if not full:
        return [], headers

    rows = [["Full reward (baseline)",
             f"{full['reward_mean']:.3f}",
             f"{full['risk_accuracy']:.3f}",
             f"{full['overreaching_accuracy']:.3f}",
             "—"]]

    for name, r in eval_results.items():
        if not name.startswith("ablation_") or name == "ablation_none":
            continue
        delta = r["reward_mean"] - full["reward_mean"]
        rows.append([
            name.replace("ablation_", "").replace("_", " "),
            f"{r['reward_mean']:.3f}",
            f"{r['risk_accuracy']:.3f}",
            f"{r['overreaching_accuracy']:.3f}",
            f"{delta:+.3f}",
        ])

    return rows, headers


def seed_variance_table(eval_results):
    """Confidence intervals across random seeds."""
    headers = ["Metric", "Mean", "Std", "Min", "Max", "n seeds"]

    seed_results = {k: v for k, v in eval_results.items() if "seed" in k}
    if not seed_results:
        return [], headers

    metrics = ["reward_mean", "risk_accuracy", "overreaching_accuracy"]
    rows = []
    for metric in metrics:
        values = [r[metric] for r in seed_results.values() if metric in r]
        if not values:
            continue
        rows.append([
            metric.replace("_", " "),
            f"{statistics.mean(values):.3f}",
            f"{statistics.stdev(values):.3f}" if len(values) > 1 else "—",
            f"{min(values):.3f}",
            f"{max(values):.3f}",
            len(values),
        ])

    return rows, headers


def hyperparameter_table(eval_results):
    """Learning rate sensitivity."""
    headers = ["Learning rate", "Reward", "Risk Acc", "Over Acc"]

    lr_results = {}
    for name, r in eval_results.items():
        m = re.search(r"lr([\d\.e\-]+)", name)
        if m:
            lr_results[float(m.group(1))] = r

    rows = []
    for lr in sorted(lr_results.keys()):
        r = lr_results[lr]
        rows.append([
            f"{lr:.0e}",
            f"{r['reward_mean']:.3f}",
            f"{r['risk_accuracy']:.3f}",
            f"{r['overreaching_accuracy']:.3f}",
        ])

    return rows, headers


def main():
    print("=" * 60)
    print("  Compiling paper results")
    print("=" * 60)

    eval_results  = load_json("./evaluation_results.json", {})
    judge_results = load_json("./llm_judge_results/summary.json", {})

    # evaluate_all.py now wraps models under "models" with a
    # "reference_reward_ceiling" alongside; support both shapes.
    ref_ceiling = None
    if "models" in eval_results:
        ref_ceiling  = eval_results.get("reference_reward_ceiling")
        eval_results = eval_results["models"]
    if ref_ceiling is not None:
        print(f"Reference reward ceiling: {ref_ceiling}")

    if not eval_results:
        print("\nNo evaluation results found.")
        print("Run: python3 evaluate_all.py")
        return

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # 1. Main comparison
    rows, headers = main_results_table(eval_results, judge_results)
    if rows:
        write_md_table(rows, headers,
                       f"{OUTPUT_DIR}/main_results_table.md",
                       "Main Results: Model Comparison")
        print(f"\n[1] Main results — {OUTPUT_DIR}/main_results_table.md")
        print(f"    | {' | '.join(headers)} |")
        for row in rows:
            print(f"    | {' | '.join(str(c) for c in row)} |")

    # 2. Ablations
    rows, headers = ablation_table(eval_results)
    if rows:
        write_md_table(rows, headers,
                       f"{OUTPUT_DIR}/ablation_table.md",
                       "Ablation Study: Reward Components")
        print(f"\n[2] Ablation table — {OUTPUT_DIR}/ablation_table.md")
        for row in rows:
            print(f"    {row}")

    # 3. Seed variance
    rows, headers = seed_variance_table(eval_results)
    if rows:
        write_md_table(rows, headers,
                       f"{OUTPUT_DIR}/seed_variance.md",
                       "Statistical Significance: Variance Across Seeds")
        print(f"\n[3] Seed variance — {OUTPUT_DIR}/seed_variance.md")
        for row in rows:
            print(f"    {row}")

    # 4. Hyperparameter sweep
    rows, headers = hyperparameter_table(eval_results)
    if rows:
        write_md_table(rows, headers,
                       f"{OUTPUT_DIR}/hyperparameter_sweep.md",
                       "Hyperparameter Sensitivity: Learning Rate")
        print(f"\n[4] Hyperparameter sweep — {OUTPUT_DIR}/hyperparameter_sweep.md")
        for row in rows:
            print(f"    {row}")

    # 5. Save consolidated JSON for figure-making
    consolidated = {
        "evaluation_results": eval_results,
        "judge_results":      judge_results,
    }
    with open(f"{OUTPUT_DIR}/all_results.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\n[5] Consolidated JSON — {OUTPUT_DIR}/all_results.json")

    # 6. Training curves (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        # Find all training logs in checkpoint directories
        log_files = glob.glob("./grpo_checkpoint*/training_log.json") + \
                    glob.glob("./grpo_checkpoint*/trainer_state.json")
        if log_files:
            print(f"\n[6] Training curves — found {len(log_files)} log files")
            # Placeholder — full plot generation would parse trainer_state.json
            # which has step-by-step loss and reward values
        else:
            print(f"\n[6] No training logs found for curves")
    except ImportError:
        print(f"\n[6] matplotlib not installed — skipping plots")
        print(f"    pip install matplotlib to enable")

    print(f"\n{'=' * 60}")
    print(f"  Done. Paper artifacts in {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
