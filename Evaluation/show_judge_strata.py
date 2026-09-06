#!/usr/bin/env python3
# show_judge_strata.py
# Stratified view of the LLM-judge ratings: breaks clinical accuracy,
# actionability, and clarity down by complexity tier, sport category,
# scenario type, and risk level -- so you can see WHERE briefs are weak,
# not just the overall averages.
#
# Joins each rating (which carries only `id` + `baseline`) to the test-set
# metadata (sport / scenario_type / complexity_tier / risk_level) keyed by
# the same `id`.
#
# Reads:
#   ./llm_judge_results/*_ratings.jsonl   (the per-brief judge scores)
#   ./formatted/sft_test.jsonl            (metadata: sport, tier, scenario)
#
# Usage:
#   python3 show_judge_strata.py                 # all trained models pooled
#   python3 show_judge_strata.py --model main    # one model
#   python3 show_judge_strata.py --by sport      # choose the slice dimension
#   python3 show_judge_strata.py --dim clinical  # focus one judge dimension

import os
import json
import glob
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

RATINGS_GLOB = "./llm_judge_results/*_ratings.jsonl"
TEST_PATHS   = ["./formatted/sft_test.jsonl", "./data_sample.jsonl"]

TRAINED = {"sft_only", "main", "ablation_none", "ablation_no_signal_conflict"}
DIMS    = ["clinical_accuracy", "actionability", "clarity"]
DIM_SHORT = {"clinical_accuracy": "Clin", "actionability": "Act", "clarity": "Clar"}


# ── Sport categorization ──────────────────────────────────────────────
# Buckets the (many) individual sports into a few meaningful categories so
# the strata have enough samples to be meaningful. Keyword-based so it
# generalizes to sports not seen in the sample.
def sport_category(sport: str) -> str:
    s = (sport or "").lower()
    endurance = ["marathon", "running", "distance", "cross_country", "skiing",
                 "cycling", "triathlon", "rowing", "swimming", "5000", "10000",
                 "800m", "1500m", "endurance"]
    strength = ["powerlifting", "weightlifting", "strength", "throw", "shot_put",
                "discus", "hammer", "sprint", "100m", "200m", "400m", "jump"]
    team = ["basketball", "football", "soccer", "rugby", "hockey", "handball",
            "netball", "volleyball", "cricket", "lacrosse", "gaelic", "baseball",
            "waterpolo", "water_polo"]
    skill = ["tennis", "table_tennis", "padel", "badminton", "gymnastics",
             "golf", "archery", "diving", "figure", "climbing", "fencing",
             "squash", "mma", "boxing", "judo", "wrestling", "taekwondo"]
    if any(k in s for k in endurance): return "endurance"
    if any(k in s for k in strength):  return "strength_power"
    if any(k in s for k in team):      return "team_field"
    if any(k in s for k in skill):     return "skill_combat"
    return "other"


def load_metadata(test_override=None):
    """id -> {sport, sport_cat, scenario_type, complexity_tier, risk_level, audience}."""
    paths = [test_override] if test_override else TEST_PATHS
    path = next((p for p in paths if p and Path(p).exists()), None)
    if path is None:
        print(f"WARNING: no metadata file found (looked for {TEST_PATHS}).")
        print("Stratified slices will be unavailable; only overall shown.")
        return {}
    meta = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # ids in ratings are the line index of the completion, which
            # matches the test-set order used to generate baselines.
            meta[i] = {
                "sport":           r.get("sport", "?"),
                "sport_cat":       sport_category(r.get("sport", "")),
                "scenario_type":   r.get("scenario_type", "?"),
                "complexity_tier": r.get("complexity_tier", "?"),
                "risk_level":      r.get("risk_level", "?"),
                "audience":        r.get("audience", "?"),
            }
    return meta, path


def load_ratings(model_filter=None):
    """Return list of rating dicts (with id, baseline, the three scores)."""
    rows = []
    for path in glob.glob(RATINGS_GLOB):
        name = Path(path).name.replace("_ratings.jsonl", "")
        if model_filter and name != model_filter:
            continue
        if model_filter is None and name not in TRAINED:
            continue   # pool only trained models by default (skip zero_shot)
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" in r or "clinical_accuracy" not in r:
                    continue
                rows.append(r)
    return rows


def mean(xs):
    return round(statistics.mean(xs), 2) if xs else None


def fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "  — "


def print_stratum_table(rows, meta, key, title):
    """Group rows by a metadata key and print mean of each judge dimension."""
    groups = defaultdict(lambda: {d: [] for d in DIMS})
    missing = 0
    for r in rows:
        m = meta.get(r["id"])
        if m is None:
            missing += 1
            continue
        g = groups[m[key]]
        for d in DIMS:
            g[d].append(r[d])

    if not groups:
        print(f"\n{title}: no data to slice (metadata join found nothing).")
        return

    print(f"\n{title}")
    print(f"  {'stratum':<20}{'Clin':<8}{'Act':<8}{'Clar':<8}{'n':<6}")
    print("  " + "-" * 46)
    # sort by clinical accuracy ascending so the weakest stratum is on top
    def sortkey(item):
        cvals = item[1]["clinical_accuracy"]
        return statistics.mean(cvals) if cvals else 99
    for stratum, g in sorted(groups.items(), key=sortkey):
        n = len(g["clinical_accuracy"])
        print(f"  {str(stratum):<20}"
              f"{fmt(mean(g['clinical_accuracy'])):<8}"
              f"{fmt(mean(g['actionability'])):<8}"
              f"{fmt(mean(g['clarity'])):<8}"
              f"{n:<6}")
    if missing:
        print(f"  ({missing} ratings had no metadata match and were skipped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Single model to analyze (default: all trained pooled)")
    ap.add_argument("--by", default="all",
                    choices=["all", "tier", "sport", "scenario", "risk", "audience"],
                    help="Which dimension to slice by")
    ap.add_argument("--test", default=None,
                    help="test metadata file (id->scenario). Use "
                         "formatted_v2/sft_test.jsonl for v2 models.")
    args = ap.parse_args()

    loaded = load_metadata(args.test)
    if isinstance(loaded, tuple):
        meta, meta_path = loaded
    else:
        meta, meta_path = loaded, None

    rows = load_ratings(args.model)
    if not rows:
        print("No ratings found. Run llm_judge.py first.")
        return

    scope = args.model if args.model else "all trained models (pooled)"
    print("=" * 60)
    print(f"  Stratified judge analysis — {scope}")
    print(f"  {len(rows)} ratings" + (f", metadata from {meta_path}" if meta_path else ""))
    print("=" * 60)

    # Overall baseline for reference
    print(f"\nOverall:")
    print(f"  {'Clin':<8}{'Act':<8}{'Clar':<8}")
    print(f"  {fmt(mean([r['clinical_accuracy'] for r in rows])):<8}"
          f"{fmt(mean([r['actionability'] for r in rows])):<8}"
          f"{fmt(mean([r['clarity'] for r in rows])):<8}")

    if not meta:
        return

    slices = {
        "tier":     ("complexity_tier", "By complexity tier (1=clear, 3=conflicting)"),
        "sport":    ("sport_cat",       "By sport category"),
        "scenario": ("scenario_type",   "By scenario type"),
        "risk":     ("risk_level",      "By ground-truth risk level"),
        "audience": ("audience",        "By audience register"),
    }
    to_show = slices.keys() if args.by == "all" else [args.by]
    for s in to_show:
        key, title = slices[s]
        print_stratum_table(rows, meta, key, title)

    print("\n(Slices sorted weakest-clinical-first. Low Clin with normal "
          "Clar = correct-looking but clinically inconsistent briefs.)")


if __name__ == "__main__":
    main()
