#!/usr/bin/env python3
# label_agreement.py
#
# Rejects generated samples where the RULE-DERIVED classification materially
# contradicts the scenario's DECLARED label, and the scenario does not declare
# the conflict as intentional.
#
# WHY THIS EXISTS
# ---------------
# rule_engine._classify_overreaching() already computes `rule_class` from the
# sampled signals, compares nothing, and returns `ground_truth` with the
# comment "Mismatch logged for quality filtering". Nothing was ever logged.
#
# Consequence, observed in post_competition (declared normal_adaptation,
# conflicting_signals=False):
#
#   SIGNAL INTEGRATION: all signals converging ... consistent with
#                       non-functional overreaching, requires immediate attention
#   CURRENT STATUS:     Your body is handling training well.
#   RECOMMENDATIONS:    Refer to sports medicine for evaluation
#
# That is not a phrasing bug. The sampler drew an NFO-level presentation under
# a normal_adaptation label. No amount of rationale prose fixes it, because the
# brief is describing a case the label does not match. The sample should never
# have been accepted.
#
# DECLARED vs UNDECLARED DIVERGENCE
#   declared (conflicting_signals=True)  -> intentional override, KEEP.
#       The rationale layer explains it (see conflict_rationale.py).
#   undeclared (conflicting_signals=False) -> a bad draw, REJECT and regenerate.
#
# CALIBRATION NOTE
# ----------------
# The threshold is deliberately conservative: only gaps of >= 2 severity ranks
# are rejected. Adjacent-class disagreement (normal_adaptation vs functional_
# overreaching) is genuinely ambiguous in the sports-science literature and
# rejecting it would discard a large share of legitimate borderline cases.
# Run in audit mode first (see main) to see the rate before enforcing.

from typing import Dict, Optional, Tuple


# Severity ordering for the overreaching axis.
#
# NOTE: "undertraining" is rank 0, NOT a severity level. rule_class returns
# "undertraining" whenever acwr_zone == "undertraining", which is *expected*
# in taper, post-competition, and critically in overtraining syndrome (the
# depleted low-load presentation). Ranking it as severe would reject every
# legitimate OTS sample.
SEVERITY_RANK = {
    "normal_adaptation": 0,
    "undertraining": 0,
    "functional_overreaching": 1,
    "non_functional_overreaching": 2,
    "overtraining_syndrome": 3,
}

# Reject when the rule-derived class differs from the declared label by at
# least this many severity ranks.
DEFAULT_MAX_GAP = 2


def derive_rule_class(acwr_zone: str,
                      hrv_days: int,
                      wellness_status: str) -> str:
    """Rule-derived overreaching class from the sampled signals.

    Mirrors rule_engine._classify_overreaching's internal `rule_class` logic
    exactly. Duplicated here so the filter can run standalone; if that logic
    changes, change it here too (or better: have rule_engine return both and
    delete this function).
    """
    if wellness_status == "severely_depressed" and hrv_days >= 21:
        return "overtraining_syndrome"
    if (acwr_zone in ("danger", "extreme")
            and hrv_days >= 7
            and wellness_status in ("moderately_depressed",
                                    "severely_depressed")):
        return "non_functional_overreaching"
    if (acwr_zone in ("caution", "danger", "extreme")
            and (hrv_days >= 3
                 or wellness_status in ("mildly_depressed",
                                        "moderately_depressed"))):
        return "functional_overreaching"
    if acwr_zone == "undertraining":
        return "undertraining"
    return "normal_adaptation"


def check_label_agreement(declared_label: str,
                          rule_class: str,
                          scenario_declares_conflict: bool,
                          max_gap: int = DEFAULT_MAX_GAP
                          ) -> Tuple[bool, Optional[str]]:
    """Return (accept: bool, reason: Optional[str]).

    Args:
        declared_label: scenario.overreaching_class
        rule_class:     derived from the sampled signals
        scenario_declares_conflict: scenario.conflicting_signals
        max_gap: severity ranks of disagreement tolerated
    """
    # Declared overrides are intentional; the rationale layer explains them.
    if scenario_declares_conflict:
        return True, None

    lab = SEVERITY_RANK.get(declared_label, 0)
    rule = SEVERITY_RANK.get(rule_class, 0)
    gap = abs(rule - lab)

    if gap >= max_gap:
        return False, (
            f"undeclared label divergence: signals imply '{rule_class}' "
            f"(rank {rule}) but scenario declares '{declared_label}' "
            f"(rank {lab}); gap {gap} >= {max_gap}"
        )
    return True, None


def check_sample(example: Dict,
                 scenario,
                 acwr_metrics: Dict,
                 hrv_analysis: Dict,
                 wellness_analysis: Dict,
                 max_gap: int = DEFAULT_MAX_GAP
                 ) -> Tuple[bool, Optional[str]]:
    """Convenience wrapper for use inside the generation loop.

    Call alongside the existing validator/quality_filter in main_parallel.py:

        ok, reason = check_sample(example, scenario_config,
                                  acwr_metrics, hrv_analysis, wellness_analysis)
        if not ok:
            rejected += 1
            continue
    """
    rule_class = derive_rule_class(
        acwr_metrics.get("zone", "sweet_spot"),
        hrv_analysis.get("consecutive_suppressed_days", 0),
        wellness_analysis.get("composite_status", "good"),
    )
    return check_label_agreement(
        getattr(scenario, "overreaching_class", "normal_adaptation"),
        rule_class,
        bool(getattr(scenario, "conflicting_signals", False)),
        max_gap,
    )


# ── Audit mode ────────────────────────────────────────────────────────
# Run over an EXISTING dataset to see how many samples this would reject
# before you enable it in generation.
#
#   python3 label_agreement.py --data dataset/train.jsonl
#
# The raw dataset does not store hrv_days / wellness_status as fields, so this
# audit infers them from the brief text. Rates here are approximate; the
# in-generator check uses the real metric values.

if __name__ == "__main__":
    import re
    import json
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/train.jsonl")
    ap.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    def infer_hrv_days(text: str) -> int:
        m = re.search(r"for (\d+) consecutive days", text or "")
        if m:
            return int(m.group(1))
        m = re.search(r"very low for (\d+) days", text or "")
        return int(m.group(1)) if m else 0

    def infer_wellness(text: str) -> str:
        low = (text or "").lower()
        for s in ("severely depressed", "moderately depressed",
                  "mildly depressed"):
            if s in low:
                return s.replace(" ", "_")
        return "good"

    n = rejected = 0
    by_scenario = Counter()
    totals = Counter()
    examples = []

    for line in open(args.data):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        scen = r.get("scenario_type", "?")
        totals[scen] += 1

        text = r.get("output_sports_scientist") or r.get("output", "")
        rule_class = derive_rule_class(
            r.get("acwr_zone", "sweet_spot"),
            infer_hrv_days(text),
            infer_wellness(text),
        )
        ok, reason = check_label_agreement(
            r.get("overreaching_classification", "normal_adaptation"),
            rule_class,
            bool(r.get("conflicting_signals", False)),
            args.max_gap,
        )
        if not ok:
            rejected += 1
            by_scenario[scen] += 1
            if len(examples) < args.show:
                examples.append((scen, reason, text[:300]))

    print("=" * 66)
    print(f"  Label-agreement audit — {args.data}  (max_gap={args.max_gap})")
    print("=" * 66)
    pct = (rejected / n * 100) if n else 0
    print(f"\nwould reject {rejected} / {n} ({pct:.1f}%)\n")
    print(f"{'scenario':<32}{'reject':<9}{'n':<8}{'rate'}")
    print("-" * 56)
    for scen in sorted(totals, key=lambda s: -(by_scenario[s] / max(totals[s], 1))):
        rate = by_scenario[scen] / max(totals[scen], 1)
        print(f"{scen:<32}{by_scenario[scen]:<9}{totals[scen]:<8}{rate:.0%}")

    if examples:
        print("\nexamples:")
        for scen, reason, snip in examples:
            print(f"\n--- {scen} ---\n  {reason}\n  {snip[:200]}...")

    print("\nA high rejection rate in one scenario usually means its declared")
    print("label and its sampling parameters disagree systematically — fix the")
    print("scenario definition rather than discarding the samples.")
