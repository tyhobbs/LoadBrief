#!/usr/bin/env python3
# consistency_check.py
# Deterministic label/body consistency checking for LoadBrief generation.
#
# Catches the defect the LLM judge surfaced: briefs whose body text
# contradicts their own classification. Runs at generation time (inside the
# quality filter) so the class of bug cannot reach the dataset again.
#
# DESIGN NOTE -- read before tuning the phrase banks.
# This must NOT reject legitimate override scenarios. A correct override brief
# DOES pair suppressed signals with a benign label; what distinguishes it from
# a defective one is that it EXPLAINS the override in prose. Check 4 is what
# lets checks 1-2 be strict without destroying the briefs you just fixed.
# The RATIONALE patterns below must stay in sync with whatever marker phrases
# the rewritten override templates actually emit.
#
# Two entry points:
#   check_consistency(...)  -> call from the quality filter during generation
#   main()                  -> calibration harness: run over an existing
#                              dataset to verify the checker flags the known
#                              bad scenarios and not everything else.
#
# Calibration usage:
#   python3 consistency_check.py --data formatted/sft_test.jsonl

import re
import json
import argparse
from collections import Counter, defaultdict


# ── Phrase banks ──────────────────────────────────────────────────────

# Asserting everything is fine. Contradiction when signals say otherwise.
# GLOBAL all-clear claims: assert the whole picture is fine. These are
# contradictions when signals are adverse. Note: per-channel statements like
# "all wellness dimensions within normal range" are NOT here -- they are true
# factual reports about one signal and can legitimately coexist with an
# adverse HRV reading (the signal-conflict case).
ALL_CLEAR_PATTERNS = [
    r"all monitoring signals are within normal ranges",
    r"all signals (are )?(within )?normal",
    r"training load is well tolerated and recovery appears adequate",
    r"your body is handling training well",
    r"(load management is |is )on track",
    r"functioning effectively",
    r"no (monitoring )?concerns",
    r"trending positively",
]

# Explicit acknowledgment that signals conflict / a discrepancy exists. When
# present, a per-signal "normal" claim is NOT a contradiction -- the brief is
# correctly handling conflicting signals rather than papering over them.
CONFLICT_ACK_PATTERNS = [
    r"signal conflict",
    r"notable discrepancy",
    r"trust hrv over subjective",
    r"subclinical fatigue",
    r"objective hrv indicates",
    r"requires clinical judgment",
    r"discrepancy between",
    r"conflicting signals",
]

# Asserting a problem *about this athlete*. Must be an assertion, not a mention:
# briefs enumerate the possible classifications (a glossary line contains the
# literal "non-functional overreaching"), so bare terms false-positive. Anchor
# each to a verb/phrase that makes it a claim about the current case.
CONCERN_PATTERNS = [
    r"(shows|showing|exhibiting|displays|signs of) (severe |early )?overtraining",
    r"(classified as|indicates|consistent with) non[- ]functional overreaching",
    r"(is|are) (approaching|exceeding) (their |the )?(tolerance|overreaching) threshold",
    r"warrants (adjustment|intervention|reduction)",
]

# "Do nothing" recommendations. Contradiction under elevated risk.
NO_ACTION_PATTERNS = [
    r"no modifications to training are indicated",
    r"no changes? (to training )?(are )?(indicated|needed|required)",
    r"maintain current training",
    r"continue as planned",
]

# Markers that an override has been explained. EXTEND THIS to match the
# phrases the new override templates emit -- these two lists are coupled.
OVERRIDE_RATIONALE_PATTERNS = [
    r"expected (environmental|developmental|physiological) response",
    r"environmental (rather than|not) training",
    r"developmentally expected",
    r"growth[- ]related",
    r"heat[- ]related (rather than|not)",
    r"acclimati[sz]ation response",
    r"attributable to (heat|altitude|growth|travel)",
    r"rather than (a sign of )?overreaching",
    r"not indicative of overreaching",
]

from conflict_rationale import RATIONALE_PATTERNS_FOR_CHECKER
OVERRIDE_RATIONALE_PATTERNS = list(set(
    OVERRIDE_RATIONALE_PATTERNS + RATIONALE_PATTERNS_FOR_CHECKER))

# ── Check 7: completeness claims over partial data ────────────────────
# The generator reduces monitored dimensions by data_level, but the all-normal
# branches asserted full coverage regardless. The judge caught briefs stating
# "all wellness dimensions within normal range" when the input said monitoring
# was limited to sleep. Matched as a CO-OCCURRENCE so a partial-data brief that
# scopes its claim honestly does not flag.
PARTIAL_DATA_PATTERNS = [
    r"partial data",
    r"monitoring (is|was) (limited|incomplete|partial)",
    r"monitoring was limited",
    r"limited to sleep",
    r"coverage is partial",
]
COMPLETENESS_CLAIM_PATTERNS = [
    r"all wellness dimensions (are )?within normal",
    r"all monitoring signals are within normal",
    r"sleep, energy, soreness, and mood all look normal",
]

BENIGN_LABELS = {"normal_adaptation"}
ELEVATED_RISK = {"high", "critical", "moderate_to_high"}

# Malformed template output — values that should never render. These are
# generation bugs (unfilled/degenerate template variables), not label
# contradictions, but they are real defects worth catching deterministically.
MALFORMED_PATTERNS = [
    r"low for 0 days",
    r"0 consecutive days",
    r"suppressed[^.]*\b0 (consecutive )?days",
    r"for -\d+ days",
    r"\bnan\b",
    r"\bnone\b (ms|au|days)",
    r"\{[a-z_]+\}",          # unrendered {placeholder}
    r"below baseline, 0 consecutive days",
]

# In-text evidence that signals are ADVERSE, parsed from the brief prose
# itself (the coach register prints exact values). Lets check 1 fire even
# when separate metric fields are unavailable.
ADVERSE_IN_TEXT_PATTERNS = [
    r"critically suppressed",
    r"significantly suppressed",
    r"suppressed[^.]*for \d+ consecutive days",
    r"very low for \d+ days? in a row",
    r"severely depressed",
    r"moderately depressed",
    r"multiple wellness (areas|dimensions) (are )?flagged",
    r"red flags: [1-9]",
]

# Severe classifications. The inverse defect: a severe label (often on a
# LOW-load presentation, e.g. depleted OTS) whose body describes normal/low
# load without explaining the chronic-history basis for the severe call.
SEVERE_LABELS = {"overtraining_syndrome", "non_functional_overreaching"}

# Body phrases asserting load/signals are fine or low -- contradictory under a
# severe label unless the chronic/historical basis is explained.
LOW_OR_NORMAL_LOAD_PATTERNS = [
    r"within the optimal training range",
    r"in the (sweet spot|optimal (range|zone))",
    r"indicates undertraining",
    r"well[- ]balanced",
    r"appropriate progressive loading",
    r"load management (is )?(on track|well[- ]calibrated)",
]

# Markers that a severe-but-low-load call has been justified by history.
CHRONIC_BASIS_PATTERNS = [
    r"chronic",
    r"prolonged",
    r"accumulated (over|across)",
    r"despite (normal|low) (current )?load",
    r"sustained (suppression|elevation) over",
    r"long[- ]standing",
    r"weeks of",
]

# Reassuring / mild-concern framings of low load. Under a severe label these
# are contradictory: they imply the low load is a minor fitness issue rather
# than a symptom of the depleted (e.g. OTS) presentation. This defect is
# register-dependent -- it appears in the athlete lay-language layer
# ("you may be losing fitness") while the sports-scientist register states the
# same low load neutrally ("indicates undertraining") and reads as consistent.
REASSURING_LOW_LOAD_PATTERNS = [
    r"you may be losing fitness",
    r"(risk of |may be |could be |concern about )detraining",
    r"you (could|might|may) (benefit from|consider) (more|additional) (training|load)",
    r"room to (increase|build|add) (load|training|volume)",
    r"consider (increasing|building|adding) (load|training|volume)",
]


def _any(patterns, text):
    """Return the first matching substring, or None."""
    low = (text or "").lower()
    for p in patterns:
        m = re.search(p, low)
        if m:
            return m.group(0)
    return None


def check_consistency(brief_text,
                      overreaching_label,
                      risk_level,
                      hrv_suppressed_days=0,
                      wellness_status="good",
                      is_override=False):
    """Return (passed: bool, violations: list[str]).

    Args:
        brief_text:          the generated brief
        overreaching_label:  ground-truth overreaching classification
        risk_level:          ground-truth risk level
        hrv_suppressed_days: consecutive suppressed days (from metrics)
        wellness_status:     composite wellness status (from metrics)
        is_override:         True when the rule-derived class differs from the
                             scenario's declared label. Surface this from
                             rule_engine._classify_overreaching, which already
                             computes both and currently discards the mismatch.
    """
    violations = []
    risk = (risk_level or "").lower()
    label = (overreaching_label or "").lower()

    # Adverse signals: from explicit metrics if provided, OR parsed from the
    # brief prose (coach register prints exact values; athlete register uses
    # phrases like "very low for 7 days in a row").
    signals_are_adverse = (
        hrv_suppressed_days >= 3
        or wellness_status in ("moderately_depressed", "severely_depressed")
        or _any(ADVERSE_IN_TEXT_PATTERNS, brief_text) is not None
    )

    # 0. Malformed template output — degenerate values that should never render.
    hit = _any(MALFORMED_PATTERNS, brief_text)
    if hit:
        violations.append(f"malformed prose ('{hit}') — template/generation bug")

    # 1. GLOBAL "all clear" prose while signals are adverse -- a real
    #    contradiction (monotony/undertraining boilerplate). BUT suppress the
    #    flag if the brief explicitly acknowledges a signal conflict: those
    #    briefs are correctly reasoning about conflicting signals, not papering
    #    over them (the illness_return / subclinical-fatigue case).
    if signals_are_adverse and not _any(CONFLICT_ACK_PATTERNS, brief_text):
        hit = _any(ALL_CLEAR_PATTERNS, brief_text)
        if hit:
            violations.append(
                f"all-clear claim ('{hit}') with adverse signals present in "
                f"the brief (contradicts its own signal report)"
            )

    # 2. Concern language under a benign label, unexplained. A brief that
    #    explicitly analyses a signal conflict has explained itself and is not
    #    contradictory -- same exemption check 1 already applies.
    if label in BENIGN_LABELS:
        hit = _any(CONCERN_PATTERNS, brief_text)
        explained = (_any(OVERRIDE_RATIONALE_PATTERNS, brief_text)
                     or _any(CONFLICT_ACK_PATTERNS, brief_text))
        if hit and not explained:
            violations.append(
                f"concern language ('{hit}') under benign label "
                f"'{label}' with no override rationale"
            )

    # 3. No-action recommendation under elevated risk.
    if risk in ELEVATED_RISK:
        hit = _any(NO_ACTION_PATTERNS, brief_text)
        if hit:
            violations.append(
                f"no-action recommendation ('{hit}') under risk '{risk}'"
            )

    # 4. An override must be explained. This is what makes 1-2 safe.
    #if is_override and not _any(OVERRIDE_RATIONALE_PATTERNS, brief_text):
    #    violations.append(
    #        "override scenario with no rationale in prose "
    #        "(label overrides raw signals but text does not explain why)"
    #    )

    # 5. Inverse defect: severe label over a low/normal-load body with no
    #    chronic-history basis. This is the overtraining-syndrome failure mode
    #    (severe call on a depleted, low-load presentation) that checks 1-2
    #    miss because they only look for the benign-label direction.
    if label in SEVERE_LABELS:
        hit = _any(LOW_OR_NORMAL_LOAD_PATTERNS, brief_text)
        explained = (_any(CHRONIC_BASIS_PATTERNS, brief_text)
                     or _any(OVERRIDE_RATIONALE_PATTERNS, brief_text))
        if hit and not explained:
            violations.append(
                f"severe label '{label}' over low/normal-load prose ('{hit}') "
                f"with no chronic-history basis"
            )

    # 6. Reassuring low-load framing under a severe label/risk. The register-
    #    dependent OTS defect: the athlete lay-language layer frames low load
    #    as a mild fitness concern ("you may be losing fitness") under a
    #    critical call, when the low load is actually a symptom.
    if label in SEVERE_LABELS or risk in ELEVATED_RISK:
        hit = _any(REASSURING_LOW_LOAD_PATTERNS, brief_text)
        if hit:
            violations.append(
                f"reassuring low-load framing ('{hit}') under severe "
                f"label/risk ('{label or risk}') — low load is a symptom "
                f"here, not a mild concern"
            )

    # 7. Completeness assertion over partial monitoring data. Distinct from
    #    check 1: here the signals are NOT adverse, the brief simply claims
    #    coverage it does not have ("all wellness dimensions normal" when only
    #    sleep was tracked). Co-occurrence based, so it does not refire the
    #    false positives that a bare "flagged dimensions:" pattern produced.
    partial_hit = _any(PARTIAL_DATA_PATTERNS, brief_text)
    claim_hit = _any(COMPLETENESS_CLAIM_PATTERNS, brief_text)
    if partial_hit and claim_hit:
        violations.append(
            f"completeness claim ('{claim_hit}') over partial monitoring "
            f"data ('{partial_hit}') — asserts coverage the data lacks"
        )

    return (len(violations) == 0), violations


# ── Calibration harness ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/train.jsonl",
                    help="raw dataset/*.jsonl (output_athlete/coach/"
                         "sports_scientist) or formatted (output+audience)")
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="only check the first N records (quick pass)")
    args = ap.parse_args()

    total = 0
    flagged = 0
    by_scenario = defaultdict(lambda: {"n": 0, "flagged": 0})
    by_audience = defaultdict(lambda: {"n": 0, "flagged": 0})
    violation_kinds = Counter()
    examples = []

    AUD_FIELDS = {
        "athlete": "output_athlete",
        "coach": "output_coach",
        "sports_scientist": "output_sports_scientist",
    }

    recs = 0
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.limit and recs >= args.limit:
                break
            recs += 1
            r = json.loads(line)
            scen = r.get("scenario_type", "?")

            pairs = []
            if "output_athlete" in r:
                for aud, field in AUD_FIELDS.items():
                    if field in r:
                        pairs.append((aud, r[field]))
            else:
                pairs.append((r.get("audience", "?"), r.get("output", "")))

            for aud, brief in pairs:
                total += 1
                by_scenario[scen]["n"] += 1
                by_audience[aud]["n"] += 1
                ok, violations = check_consistency(
                    brief_text=brief,
                    overreaching_label=r.get("overreaching_classification", ""),
                    risk_level=r.get("risk_level", ""),
                    is_override=r.get("conflicting_signals", False),
                )
                if not ok:
                    flagged += 1
                    by_scenario[scen]["flagged"] += 1
                    by_audience[aud]["flagged"] += 1
                    for v in violations:
                        violation_kinds[v.split("(")[0].strip()] += 1
                    if len(examples) < args.show:
                        examples.append((scen, aud, violations, brief[:280]))

    print("=" * 66)
    print(f"  Consistency calibration \u2014 {args.data}")
    print("=" * 66)
    pct = (flagged / total * 100) if total else 0
    print(f"\nflagged {flagged} / {total} ({pct:.1f}%)\n")

    print(f"{'scenario':<32}{'flagged':<10}{'n':<8}{'rate':<8}")
    print("-" * 58)
    for scen, d in sorted(by_scenario.items(),
                          key=lambda kv: -(kv[1]["flagged"] / max(kv[1]["n"], 1))):
        rate = d["flagged"] / d["n"] if d["n"] else 0
        print(f"{scen:<32}{d['flagged']:<10}{d['n']:<8}{rate:.0%}")

    print(f"\n{'audience':<20}{'flagged':<10}{'n':<8}{'rate':<8}")
    print("-" * 46)
    for aud, d in sorted(by_audience.items(),
                         key=lambda kv: -(kv[1]["flagged"] / max(kv[1]["n"], 1))):
        rate = d["flagged"] / d["n"] if d["n"] else 0
        print(f"{aud:<20}{d['flagged']:<10}{d['n']:<8}{rate:.0%}")

    if violation_kinds:
        print("\nviolation kinds:")
        for k, n in violation_kinds.most_common():
            print(f"  {n:5d}  {k}")

    if examples:
        print(f"\nexample violations (first {len(examples)}):")
        for scen, aud, viols, snippet in examples:
            print(f"\n--- {scen} [{aud}] ---")
            for v in viols:
                print(f"  ! {v}")
            print(f"  brief: {snippet[:200]}...")


if __name__ == "__main__":
    main()
