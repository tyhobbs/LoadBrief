#!/usr/bin/env python3
# show_judge.py
# Print the rule-based metrics (from evaluate_all.py) next to the
# LLM-as-judge scores (from llm_judge.py) in one side-by-side table.
#
# Reads:
#   ./evaluation_results.json      (reward, risk acc, within-1, overreaching)
#   ./llm_judge_results/summary.json   (clinical / action / clarity, 1-7)
#
# Either file may be missing or partial; whatever is present is shown.
# Usage:  python3 show_judge.py

import json
import glob
import statistics
from pathlib import Path

EVAL_PATH    = "./evaluation_results.json"
SUMMARY_PATH = "./llm_judge_results/summary.json"
RATINGS_GLOB = "./llm_judge_results/*_ratings.jsonl"


def load_eval():
    """Return {model: {reward, risk, within1, over}} and the ceiling."""
    if not Path(EVAL_PATH).exists():
        return {}, None
    with open(EVAL_PATH) as f:
        data = json.load(f)
    ceiling = None
    if "models" in data:                      # new wrapped shape
        ceiling = data.get("reference_reward_ceiling")
        data = data["models"]
    out = {}
    for name, m in data.items():
        out[name] = {
            "reward":  m.get("reward_mean"),
            "risk":    m.get("risk_accuracy"),
            "within1": m.get("risk_within_one"),
            "over":    m.get("overreaching_accuracy"),
        }
    return out, ceiling


def load_judge():
    """Return {model: {clinical, action, clarity, overall, n}}.

    Prefer summary.json; if missing, rebuild from the per-model
    *_ratings.jsonl files (ignoring any error rows)."""
    judge = {}
    if Path(SUMMARY_PATH).exists():
        try:
            with open(SUMMARY_PATH) as f:
                s = json.load(f)
            for name, m in s.items():
                judge[name] = {
                    "clinical": m.get("clinical_accuracy_mean"),
                    "action":   m.get("actionability_mean"),
                    "clarity":  m.get("clarity_mean"),
                    "overall":  m.get("overall_mean"),
                    "n":        m.get("n_rated"),
                }
        except Exception:
            judge = {}

    # Fill in any models missing from summary.json straight from ratings files
    for path in glob.glob(RATINGS_GLOB):
        name = Path(path).name.replace("_ratings.jsonl", "")
        if name in judge:
            continue
        ca, ac, cl = [], [], []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" not in r and "clinical_accuracy" in r:
                    ca.append(r["clinical_accuracy"])
                    ac.append(r["actionability"])
                    cl.append(r["clarity"])
        if ca:
            judge[name] = {
                "clinical": round(statistics.mean(ca), 2),
                "action":   round(statistics.mean(ac), 2),
                "clarity":  round(statistics.mean(cl), 2),
                "overall":  round(statistics.mean(ca + ac + cl), 2),
                "n":        len(ca),
            }
    return judge


def cell(v, fmt="{:.3f}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "—"


def main():
    eval_m, ceiling = load_eval()
    judge_m = load_judge()

    models = sorted(set(eval_m) | set(judge_m))
    if not models:
        print("No results found. Run evaluate_all.py and/or llm_judge.py first.")
        return

    # Sort by judge overall if available, else by reward
    def sort_key(name):
        j = judge_m.get(name, {})
        e = eval_m.get(name, {})
        return (-(j.get("overall") or 0), -(e.get("reward") or 0))
    models.sort(key=sort_key)

    if ceiling is not None:
        print(f"Reference reward ceiling: {ceiling}  "
              f"(read Reward relative to this, not 1.0)\n")

    header = (f"{'Model':<26}"
              f"{'Reward':<9}{'RiskAcc':<9}{'W-in1':<8}{'OverAcc':<9}"
              f"| {'Clin':<7}{'Act':<7}{'Clar':<7}{'Ovr':<7}{'n':<5}")
    print(header)
    print("-" * len(header))

    for name in models:
        e = eval_m.get(name, {})
        j = judge_m.get(name, {})
        print(f"{name:<26}"
              f"{cell(e.get('reward')):<9}{cell(e.get('risk')):<9}"
              f"{cell(e.get('within1')):<8}{cell(e.get('over')):<9}"
              f"| {cell(j.get('clinical'),'{:.2f}'):<7}"
              f"{cell(j.get('action'),'{:.2f}'):<7}"
              f"{cell(j.get('clarity'),'{:.2f}'):<7}"
              f"{cell(j.get('overall'),'{:.2f}'):<7}"
              f"{cell(j.get('n'),'{:d}'):<5}")

    print("-" * len(header))
    print("Left block: rule-based metrics (0-1).  "
          "Right block: LLM-judge (1-7).")

    # Quick read on the key question
    trained = {n: j for n, j in judge_m.items()
               if n not in ("zero_shot",) and j.get("overall") is not None}
    if len(trained) >= 2:
        vals = [j["overall"] for j in trained.values()]
        spread = max(vals) - min(vals)
        print(f"\nJudge overall spread across trained models: {spread:.2f} "
              f"(small spread => judge agrees SFT and GRPO are comparable).")


if __name__ == "__main__":
    main()
