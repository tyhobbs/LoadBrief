# evaluate_all.py
# Score every baseline output and trained model variant against the test set.
# Computes:
#   - Mean reward score (from composite reward function)
#   - Ground truth accuracy on risk level and overreaching classification
#   - Output statistics (length, vocabulary diversity)
#
# Usage:
#   python3 evaluate_all.py
# This processes everything in ./baseline_outputs/ and saves to ./evaluation_results.json

import os
import re
import json
import statistics
from pathlib import Path
from collections import Counter

BASELINE_DIR = "./baseline_outputs_v8"
TEST_DATA    = "./formatted_v8/sft_test.jsonl"
OUTPUT       = "./evaluation_results_v8.json"


def composite_reward(completion):
    """The 5-component reward function — single example version."""
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

    return min(1.0, round(score, 3))


# ── Register-aware extraction ─────────────────────────────────────────
# The three audience registers use different headers:
#   athlete:           YOUR STATUS: MODERATE
#   coach:             RISK LEVEL: MODERATE
#   sports_scientist:  OVERALL RISK CLASSIFICATION: MODERATE
# and different classification sections:
#   athlete:           CURRENT STATUS:        (plain language)
#   coach:             CLINICAL CLASSIFICATION:
#   sports_scientist:  OVERREACHING CLASSIFICATION:

RISK_HEADER_RE = re.compile(
    r'(?:overall\s+risk\s+classification|risk\s+level|your\s+status)'
    r'\s*:\s*\**\s*(low|moderate|high|critical)',
    re.IGNORECASE,
)

# Boundary/compound ground-truth labels: which extracted base classes
# count as correct. (The brief renders the boundary label as one side.)
RISK_ACCEPT = {
    "low":                      {"low"},
    "moderate":                 {"moderate"},
    "high":                     {"high"},
    "critical":                 {"critical"},
    "moderate_to_high":         {"moderate", "high"},
    "low_if_managed":           {"low"},
    "low_with_detraining_flag": {"low"},
}


def extract_predicted_risk(text):
    """Extract the declared risk level via the register headers.
    Returns 'unknown' when no header is present (e.g. unformatted
    zero-shot output) rather than guessing from stray words."""
    m = RISK_HEADER_RE.search(text or "")
    return m.group(1).lower() if m else "unknown"


_SECTION_NEXT = r'(?=^\s*[A-Z][A-Z /&()\-]{2,}:\s*$|^\s*[A-Z][A-Z /&()\-]{2,}:|\Z)'


def _section(text, header_names):
    """Return the body text of the first matching ALL-CAPS section."""
    pat = re.compile(
        r'^\s*(?:' + '|'.join(header_names) + r')\s*:\s*(.*?)' + _SECTION_NEXT,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    m = pat.search(text or "")
    return m.group(1).strip() if m else None


def extract_predicted_overreaching(text):
    """Extract the overreaching classification, anchored to the brief's
    classification section. Match order is critical:
    'normal adaptation' must be checked FIRST because the
    sports_scientist normal variant reads 'No functional or
    non-functional overreaching markers present' — checking NFO first
    would match the negation. Likewise 'non-functional' before
    'functional' (substring containment)."""
    sec = _section(text, [
        r'overreaching\s+classification',   # sports_scientist
        r'clinical\s+classification',       # coach
        r'current\s+status',                # athlete (plain language)
    ])
    scope = (sec if sec is not None else (text or "")).lower()

    # Clinical vocabulary — order matters (see docstring)
    if "normal adaptation" in scope:
        return "normal_adaptation"
    if "overtraining syndrome" in scope:
        return "overtraining_syndrome"
    if ("non-functional overreaching" in scope
            or "non functional overreaching" in scope
            or "nonfunctional overreaching" in scope):
        return "non_functional_overreaching"
    if "functional overreaching" in scope:
        return "functional_overreaching"

    # Athlete-register lay language (only trusted inside the section).
    # Exact template phrasings:
    #   OTS: "severe overtraining signs ... medical support"
    #   NFO: "overtraining warning signs that need 2-6 weeks"
    #   FO:  "early overtraining signs ... easy days will resolve"
    #   NA:  "handling training well"
    if sec is not None:
        if "severe overtraining" in scope or "medical support" in scope:
            return "overtraining_syndrome"
        if "warning signs" in scope or "weeks" in scope:
            return "non_functional_overreaching"
        if ("early overtraining" in scope
                or ("easy days" in scope and "resolve" in scope)):
            return "functional_overreaching"
        if ("handling training well" in scope or "adapting well" in scope
                or "responding well" in scope):
            return "normal_adaptation"

    return "unknown"


def _norm_risk(v):
    """Normalize a risk label: 'HIGH' / ' High ' -> 'high'."""
    return str(v).strip().lower() if v is not None else None


def _norm_over(v):
    """Normalize an overreaching label:
    'Non-Functional Overreaching' / 'non_functional_overreaching'
    -> 'non_functional_overreaching'."""
    if v is None:
        return None
    return (str(v).strip().lower()
            .replace("-", "_").replace(" ", "_"))


def load_ground_truth(test_path, n=500):
    """Load ground truth labels from the formatted test set.

    Labels live as TOP-LEVEL keys in formatted/sft_test.jsonl
    (risk_level, overreaching_classification) — not nested under
    a 'ground_truth_labels' dict.
    """
    gt = {}
    with open(test_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line.strip())
            labels = {}
            if ex.get("risk_level") is not None:
                labels["risk_level"] = _norm_risk(ex["risk_level"])
            if ex.get("overreaching_classification") is not None:
                labels["overreaching_classification"] = _norm_over(
                    ex["overreaching_classification"])
            labels["audience"] = ex.get("audience", "?")
            gt[i] = labels
    return gt


def evaluate_baseline(name, output_path, ground_truth):
    """Compute all metrics for one baseline."""
    if not Path(output_path).exists():
        return None

    results = []
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        return None

    # Compute reward for each
    rewards = [composite_reward(r["completion"]) for r in results]

    # Class ordering for error-distance analysis
    RANK = {"low": 0, "moderate": 1, "high": 2, "critical": 3}

    # Compute classification accuracy (accept-sets for boundary labels,
    # per-register breakdown, and error-distance stats for the paper)
    risk_correct = 0
    risk_total   = 0
    over_correct = 0
    over_total   = 0
    risk_within_one = 0     # correct OR adjacent class
    risk_severe     = 0     # off by 2+ classes
    risk_unknown    = 0     # no header found in completion
    per_reg = {}   # audience -> [risk_ok, risk_n, over_ok, over_n]
    for r in results:
        gt = ground_truth.get(r["id"], {})
        aud = gt.get("audience", "?")
        reg = per_reg.setdefault(aud, [0, 0, 0, 0])
        if "risk_level" in gt:
            risk_total += 1
            reg[1] += 1
            pred = _norm_risk(extract_predicted_risk(r["completion"]))
            accepted = RISK_ACCEPT.get(gt["risk_level"], {gt["risk_level"]})
            if pred in accepted:
                risk_correct += 1
                risk_within_one += 1
                reg[0] += 1
            elif pred == "unknown" or pred not in RANK:
                risk_unknown += 1
            else:
                dist = min(abs(RANK[pred] - RANK[a])
                           for a in accepted if a in RANK) if any(
                               a in RANK for a in accepted) else 99
                if dist <= 1:
                    risk_within_one += 1
                else:
                    risk_severe += 1
        if "overreaching_classification" in gt:
            over_total += 1
            reg[3] += 1
            pred = _norm_over(extract_predicted_overreaching(r["completion"]))
            if pred == gt["overreaching_classification"]:
                over_correct += 1
                reg[2] += 1

    per_register = {
        aud: {
            "risk_accuracy": round(v[0] / max(v[1], 1), 3),
            "over_accuracy": round(v[2] / max(v[3], 1), 3),
            "n":             v[1],
        }
        for aud, v in per_reg.items()
    }

    # Output stats
    word_counts = [len(r["completion"].split()) for r in results]
    nonempty    = sum(1 for r in results if r["completion"].strip())

    return {
        "baseline":              name,
        "n_examples":            len(results),
        "n_nonempty":            nonempty,
        "reward_mean":           round(statistics.mean(rewards), 3),
        "reward_std":            round(statistics.stdev(rewards) if len(rewards) > 1 else 0, 3),
        "reward_median":         round(statistics.median(rewards), 3),
        "risk_accuracy":         round(risk_correct / max(risk_total, 1), 3),
        "risk_within_one":       round(risk_within_one / max(risk_total, 1), 3),
        "risk_severe_errors":    risk_severe,
        "risk_unknown":          risk_unknown,
        "overreaching_accuracy": round(over_correct / max(over_total, 1), 3),
        "per_register":          per_register,
        "word_count_mean":       round(statistics.mean(word_counts), 1),
        "word_count_median":     int(statistics.median(word_counts)),
    }


def main():
    print("=" * 60)
    print("  Evaluating all baselines and model variants")
    print("=" * 60)

    print(f"\nLoading ground truth labels...")
    ground_truth = load_ground_truth(TEST_DATA)
    print(f"Loaded {len(ground_truth)} ground truth labels")

    # Reference reward ceiling: score the ground-truth briefs themselves.
    # Models should be read relative to this, not to a theoretical 1.0.
    ref_rewards = []
    with open(TEST_DATA, "r") as f:
        for i, line in enumerate(f):
            if i >= len(ground_truth):
                break
            ex = json.loads(line.strip())
            if ex.get("output"):
                ref_rewards.append(composite_reward(ex["output"]))
    ref_ceiling = round(statistics.mean(ref_rewards), 3) if ref_rewards else None
    if ref_ceiling is not None:
        print(f"Reference reward ceiling (ground-truth briefs): {ref_ceiling}")

    # Find all baseline output files
    output_files = sorted(Path(BASELINE_DIR).glob("*.jsonl")) if Path(BASELINE_DIR).exists() else []
    if not output_files:
        print(f"\nNo files found in {BASELINE_DIR}")
        print("Run baselines first: python3 run_baselines.py --baseline all")
        return

    print(f"\nFound {len(output_files)} baseline output files\n")

    results = {}
    for path in output_files:
        name = path.stem
        print(f"Evaluating {name}...")
        result = evaluate_baseline(name, path, ground_truth)
        if result:
            results[name] = result

    # Print comparison table
    print("\n" + "=" * 100)
    print(f"{'Baseline':<25} {'Reward':<12} {'Risk Acc':<10} {'Within-1':<10} "
          f"{'Severe':<8} {'Over Acc':<10} {'Words':<8}")
    print("=" * 100)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["reward_mean"]):
        print(f"{name:<25} {r['reward_mean']:<5}±{r['reward_std']:<5} "
              f"{r['risk_accuracy']:<10} {r['risk_within_one']:<10} "
              f"{r['risk_severe_errors']:<8} {r['overreaching_accuracy']:<10} "
              f"{int(r['word_count_mean']):<8}")
    print("=" * 100)
    if ref_ceiling is not None:
        print(f"(Reference reward ceiling: {ref_ceiling} — read the Reward "
              f"column relative to this, not to 1.0)")

    # Per-register breakdown (risk / overreaching accuracy by audience)
    auds = sorted({a for r in results.values()
                   for a in r.get("per_register", {})})
    if auds:
        print("\nPer-register accuracy (risk / overreaching):")
        header = f"{'Baseline':<25}" + "".join(f"{a:<22}" for a in auds)
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for name, r in sorted(results.items(), key=lambda x: -x[1]["reward_mean"]):
            cells = ""
            for a in auds:
                pr = r.get("per_register", {}).get(a)
                cells += (f"{pr['risk_accuracy']:.2f} / {pr['over_accuracy']:.2f}"
                          .ljust(22) if pr else "—".ljust(22))
            print(f"{name:<25}{cells}")
        print("-" * len(header))

    # Save
    output = {"reference_reward_ceiling": ref_ceiling, "models": results}
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
