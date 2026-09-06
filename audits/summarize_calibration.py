#!/usr/bin/env python3
"""summarize_calibration.py — turn calibration cells into the paper's table.

Reads every calibration/<scenario>__<audience>__<judge>.jsonl produced by
run_full_calibration.sh, computes ground-truth judge scores with bootstrap
confidence intervals, joins the fine-tuned model's scores for the same
(scenario, audience) cells, and reports the gap.

Usage:
    python3 summarize_calibration.py --calibration calibration

    # explicit paths if discovery misses them
    python3 summarize_calibration.py --calibration calibration \\
        --ratings "llm judge results full/llm_judge_results_v8/sft_v8_ratings.jsonl" \\
        --test "Formatted Datasets/formatted_v8/sft_test.jsonl"

    # also write the LaTeX table body for the paper
    python3 summarize_calibration.py --calibration calibration --latex calib_table.tex

A positive gap means the model scores BELOW its own training targets. A negative
gap means the model scores ABOVE the ground-truth briefs it was trained on --
evidence that the stratum's weakness is a property of the corpus, not the model.
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
SKIP_PREFIX = ("sft_checkpoint", "grpo_checkpoint", "final_model", "baseline_outputs")


def walk(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(SKIP_PREFIX)]
        for fn in fns:
            yield os.path.join(dp, fn)


def find(root, basename, prefer):
    cands = [f for f in walk(root) if os.path.basename(f) == basename]
    if not cands:
        return None
    cands.sort(key=lambda p: (0 if prefer in p.replace(os.sep, "/") else 1,
                              p.count(os.sep)))
    return cands[0]


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def boot_ci(vals, iters=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean."""
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(iters):
        means.append(sum(rng.choice(vals) for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="calibration")
    ap.add_argument("--root", default=".")
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--judge", default=None, help="restrict to one judge")
    ap.add_argument("--latex", default=None, help="write LaTeX table body here")
    a = ap.parse_args()

    # ---- ground-truth calibration cells ----
    gt = defaultdict(list)          # (scenario, audience, judge) -> [clinical]
    gt_all = defaultdict(lambda: defaultdict(list))
    pat = re.compile(r"^(.+?)__(athlete|coach|sports_scientist)__(\w+)\.jsonl$")
    if not os.path.isdir(a.calibration):
        sys.exit(f"no such directory: {a.calibration}")
    for fn in sorted(os.listdir(a.calibration)):
        m = pat.match(fn)
        if not m:
            continue
        scen, aud, judge = m.groups()
        if a.judge and judge != a.judge:
            continue
        for r in load_jsonl(os.path.join(a.calibration, fn)):
            if "clinical_accuracy" not in r:
                continue
            gt[(scen, aud, judge)].append(r["clinical_accuracy"])
            for d in ("clinical_accuracy", "actionability", "clarity"):
                if d in r:
                    gt_all[(scen, aud, judge)][d].append(r[d])
    if not gt:
        sys.exit(f"no calibration cells found in {a.calibration}/")

    # ---- model scores for the same cells ----
    ratings = a.ratings or find(a.root, "sft_v8_ratings.jsonl", "llm_judge_results_v8/")
    test = a.test or find(a.root, "sft_test.jsonl", "formatted_v8/")
    model = defaultdict(list)
    if ratings and test:
        meta = {}
        for i, r in enumerate(load_jsonl(test)):
            meta[i] = (r.get("scenario_type", "?"), r.get("audience", "?"))
        for r in load_jsonl(ratings):
            if "clinical_accuracy" not in r:
                continue
            k = meta.get(r["id"])
            if k:
                model[k].append(r["clinical_accuracy"])
        print(f"model scores from {os.path.relpath(ratings, a.root)}")
        print(f"joined via       {os.path.relpath(test, a.root)}\n")
    else:
        print("WARNING: model ratings/test split not found; gaps unavailable.\n")

    judges = sorted({j for _, _, j in gt})
    rows = []
    for judge in judges:
        print("=" * 96)
        print(f"  Ground-truth judge calibration — {judge}")
        print("=" * 96)
        print(f"{'scenario':30s}{'reg':10s}{'n':>3}{'GT clin':>9}"
              f"{'95% CI':>16}{'model':>8}{'n':>5}{'gap':>8}")
        print("-" * 96)
        for (scen, aud, j) in sorted(gt):
            if j != judge:
                continue
            v = gt[(scen, aud, j)]
            mu = statistics.mean(v)
            lo, hi = boot_ci(v)
            mv = model.get((scen, aud), [])
            mmu = statistics.mean(mv) if mv else float("nan")
            gap = mmu - mu if mv else float("nan")
            rows.append((judge, scen, aud, len(v), mu, lo, hi, mmu, len(mv), gap))
            gs = f"{gap:+.2f}" if mv else "   --"
            ms = f"{mmu:.2f}" if mv else "  --"
            print(f"{scen[:29]:30s}{aud[:9]:10s}{len(v):>3}{mu:>9.2f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>16}{ms:>8}{len(mv):>5}{gs:>8}")

        sub = [r for r in rows if r[0] == judge and r[8] > 0]
        if len(sub) >= 3:
            g = [r[9] for r in sub]
            print("-" * 96)
            print(f"  cells with model comparison : {len(sub)}")
            print(f"  mean gap (model - GT)       : {statistics.mean(g):+.2f}")
            print(f"  cells where model >= GT     : {sum(1 for x in g if x >= 0)}"
                  f" / {len(g)}")
            rho = spearman([r[4] for r in sub], [r[7] for r in sub])
            print(f"  Spearman rho (GT vs model)  : {rho:+.3f}")
            print("  A high rho means the model tracks its training targets stratum")
            print("  by stratum -- i.e. weak strata are weak in the DATA.")
        print()

    # ---- cross-judge agreement ----
    if len(judges) > 1:
        print("=" * 96)
        print("  Cross-judge agreement on ground-truth briefs")
        print("=" * 96)
        for d in ("clinical_accuracy", "actionability", "clarity"):
            pairs = []
            for (scen, aud, j1) in gt_all:
                if j1 != judges[0]:
                    continue
                for j2 in judges[1:]:
                    k2 = (scen, aud, j2)
                    if k2 in gt_all and gt_all[k2][d]:
                        pairs.append((statistics.mean(gt_all[(scen, aud, j1)][d]),
                                      statistics.mean(gt_all[k2][d])))
            if len(pairs) >= 2:
                diffs = [x - y for x, y in pairs]
                print(f"  {d:20s} n={len(pairs):3d}  "
                      f"mean |diff| = {statistics.mean([abs(x) for x in diffs]):.2f}  "
                      f"bias({judges[0]}-{judges[1]}) = {statistics.mean(diffs):+.2f}  "
                      f"rho = {spearman([p[0] for p in pairs], [p[1] for p in pairs]):+.3f}")
        print()

    # ---- LaTeX ----
    if a.latex:
        with open(a.latex, "w") as f:
            f.write("% generated by summarize_calibration.py\n")
            for (judge, scen, aud, n, mu, lo, hi, mmu, mn, gap) in rows:
                if not mn:
                    continue
                nm = scen.replace("_", r"\_")
                f.write(f"{nm} & {aud.replace('_', chr(92)+'_')} & {judge} & "
                        f"{mu:.2f} & [{lo:.2f}, {hi:.2f}] & {mmu:.2f} & "
                        f"${gap:+.2f}$ \\\\\n")
        print(f"LaTeX rows written to {a.latex}")


if __name__ == "__main__":
    main()
