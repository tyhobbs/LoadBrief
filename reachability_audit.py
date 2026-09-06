#!/usr/bin/env python3
"""Reachability audit — do scenarios produce the metric ranges they declare?

WHY THIS EXISTS
---------------
Generation runs backward from the label: a scenario declares a target range,
and a load pattern is supposed to produce it. Nothing checks that the pattern
CAN. Seven scenarios turned out to declare targets their own sampling could
never reach:

    acwr_spike            declared 1.5-2.0    produced ~1.05   (0% overlap)
    undertraining         declared 0.3-0.7    produced ~1.00   (0% overlap)
    high_acwr_stable      declared 1.5-1.8    produced ~1.00   (0% overlap)
    monotony_problem      declared 2.5-4.0    produced ~1.05   (0% overlap)
    early_overreaching    declared 1.3-1.7    produced ~1.00   (0% overlap)
    illness_return        declared 1.2-1.8    inflated denominator
    post_competition      declared 0.6-0.9    produced ~1.00   (0% overlap)

Each was found separately — by the label-agreement filter, by judge comments,
by a metric audit, by residual analysis — and six of the seven only surfaced
AFTER a model had already trained on the bad data. Every one was detectable in
seconds by this check.

The failure is invisible to conventional validation because the label/data
consistency check PASSES: the label is consistent with whatever was sampled.
It is the DECLARATION that is unreachable, and nothing compares the two.

WHAT IT CHECKS
--------------
For each scenario, compares the declared target range against the distribution
actually present in a generated corpus:

    overlap    fraction of samples landing inside the declared range
    p5-p95     where the samples actually land
    verdict    OK / MARGINAL / UNREACHABLE

Note this audits ACWR and monotony only — the numeric targets scenarios
declare. A scenario whose label rests on HRV days or wellness composite is not
covered; extend METRICS below if those gain declared ranges.

USAGE
  python3 reachability_audit.py --data dataset_v6/train.jsonl
  python3 reachability_audit.py --data dataset_v6/train.jsonl --json audit.json

EXIT CODE
  0 if every scenario is reachable, 1 otherwise — so it can gate a build.
"""
import re
import sys
import json
import argparse
import statistics
from collections import defaultdict

# Fraction of samples inside the declared range below which we complain.
UNREACHABLE_MAX = 0.05     # <=5% in range: the target is effectively unreachable
MARGINAL_MAX = 0.40        # <=40%: reachable but the sampler is poorly centred


def load_declared_targets(path):
    """Parse declared ranges out of scenarios.py.

    Read from source rather than importing so the audit runs without the
    generator's dependencies on the path.
    """
    src = open(path).read()
    blocks = re.split(r"\bSCENARIO_[A-Z_]+\s*=\s*ScenarioConfig\(", src)[1:]
    out = {}
    for b in blocks:
        nm = re.search(r'name\s*=\s*"([^"]+)"', b)
        if not nm:
            continue
        scen = nm.group(1)
        targets = {}
        m = re.search(r"acwr_target_final\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", b)
        if m:
            targets["acwr"] = (float(m.group(1)), float(m.group(2)))
        m = re.search(r'"monotony_index_target"\s*:\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', b)
        if m:
            targets["monotony"] = (float(m.group(1)), float(m.group(2)))
        if targets:
            out[scen] = targets
    return out


def collect_produced(path):
    """Gather the metric values actually present in a generated corpus.

    ACWR is a stored field. Monotony is only reported in the narrative, so it
    is parsed back out of the text.
    """
    acwr = defaultdict(list)
    mono = defaultdict(list)
    mono_pat = re.compile(r"monotony[^0-9]{0,30}([0-9.]+)", re.I)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            scen = r.get("scenario_type")
            if not scen:
                continue
            v = r.get("acwr_value")
            if isinstance(v, (int, float)):
                acwr[scen].append(float(v))
            m = mono_pat.search(r.get("input_narrative", "") or "")
            if m:
                try:
                    mono[scen].append(float(m.group(1)))
                except ValueError:
                    pass
    return acwr, mono


def assess(vals, lo, hi):
    if not vals:
        return None
    inside = sum(1 for v in vals if lo <= v <= hi) / len(vals)
    s = sorted(vals)
    p5 = s[max(0, int(0.05 * len(s)) - 1)]
    p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    if inside <= UNREACHABLE_MAX:
        verdict = "UNREACHABLE"
    elif inside <= MARGINAL_MAX:
        verdict = "MARGINAL"
    else:
        verdict = "OK"
    return {"n": len(vals), "overlap": round(inside, 3),
            "p5": round(p5, 3), "p95": round(p95, 3),
            "median": round(statistics.median(vals), 3),
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="generated corpus, e.g. dataset_v6/train.jsonl")
    ap.add_argument("--scenarios",
                    default="loadbrief_generator/simulator/scenarios.py")
    ap.add_argument("--json", default=None, help="write results here")
    a = ap.parse_args()

    declared = load_declared_targets(a.scenarios)
    acwr, mono = collect_produced(a.data)
    produced = {"acwr": acwr, "monotony": mono}

    print("=" * 92)
    print("  REACHABILITY AUDIT — do scenarios produce the ranges they declare?")
    print(f"  corpus: {a.data}")
    print("=" * 92)
    print(f"  {'scenario':<30}{'metric':<10}{'declared':<14}"
          f"{'produced p5-p95':<20}{'in range':<10}verdict")
    print("-" * 92)

    results = {}
    bad = []
    for scen in sorted(declared):
        for metric, (lo, hi) in sorted(declared[scen].items()):
            vals = produced[metric].get(scen, [])
            r = assess(vals, lo, hi)
            if r is None:
                print(f"  {scen:<30}{metric:<10}{f'{lo}-{hi}':<14}"
                      f"{'(no data)':<20}{'—':<10}SKIPPED")
                continue
            results.setdefault(scen, {})[metric] = dict(
                r, declared=[lo, hi])
            flag = {"OK": "", "MARGINAL": "  <-- poorly centred",
                    "UNREACHABLE": "  <-- FIX THIS"}[r["verdict"]]
            span = f"{r['p5']}-{r['p95']}"
            print(f"  {scen:<30}{metric:<10}{f'{lo}-{hi}':<14}"
                  f"{span:<20}{r['overlap']:<10.0%}{r['verdict']}{flag}")
            if r["verdict"] != "OK":
                bad.append((scen, metric, r))

    print("-" * 92)
    if not bad:
        print("  All declared targets are reachable.")
    else:
        print(f"  {len(bad)} target(s) need attention:\n")
        for scen, metric, r in bad:
            lo, hi = results[scen][metric]["declared"]
            print(f"    {scen} / {metric}")
            print(f"      declares {lo}-{hi}, produces {r['p5']}-{r['p95']} "
                  f"(median {r['median']}), {r['overlap']:.0%} in range")
            if r["verdict"] == "UNREACHABLE":
                print("      The load pattern cannot produce this range. Either the")
                print("      pattern or the declared target is wrong — decide which,")
                print("      because the label currently rests on data that never")
                print("      supports it.")
            print()

    print("\n  NOTE: ACWR and monotony only. Scenarios whose labels rest on HRV")
    print("  or wellness thresholds are not covered by this audit.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"corpus": a.data, "results": results}, f, indent=2)
        print(f"\n  wrote {a.json}")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
