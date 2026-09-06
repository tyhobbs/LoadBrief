#!/usr/bin/env python3
"""verify_paper_tables.py — recompute every verifiable number in the LoadBrief
paper from source data and diff it against the published value.

Point it at the project root and it finds everything itself, at any depth:

    python3 verify_paper_tables.py --root ~/LoadBrief

It searches recursively for the corpus, the evaluation JSONs, the judge
ratings, the formatted test split, and loadbrief_generator/, so it does not
care how the folder is organized. Files it picks are printed up front, with
alternatives listed when a choice was ambiguous.

Target a different revision with --version (default v8):

    python3 verify_paper_tables.py --root ~/LoadBrief --version v8

Any path can still be pinned explicitly, which overrides discovery:

    python3 verify_paper_tables.py --root ~/LoadBrief \
        --train dataset_v8/train.jsonl \
        --test  formatted_v8/sft_test.jsonl

Add --check-validator to measure the schema-validator row of Table IV instead
of asserting it. Every check prints PASS or FAIL with expected and actual
values; exit code 0 if all pass, 1 otherwise.

WHAT THIS DOES NOT COVER
------------------------
Table V (reachability): those figures came from the reachability_audit.py
docstring and describe earlier corpora. Run reachability_audit.py against each
surviving dataset_v*/train.jsonl to verify, or mark them as historical.

Table IV consistency-checker and label-agreement rows: run consistency_check.py
and label_agreement.py directly.
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

# Directories never worth walking: model weights, caches, venvs, raw completions.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".ipynb_checkpoints",
    "node_modules", ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
    "wandb", ".cache", ".DS_Store",
}
SKIP_DIR_PREFIXES = ("sft_checkpoint", "grpo_checkpoint", "final_model",
                     "checkpoint-", "baseline_outputs")


def walk_project(root):
    """Yield every file under root, skipping weight and cache directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(SKIP_DIR_PREFIXES)
        ]
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def _looks_like_corpus(path):
    """True if the first line has the corpus fields, so we don't pick up a
    same-named file from a checkpoint or an unrelated split."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                return "input_narrative" in r and "scenario_type" in r
    except Exception:
        return False
    return False


def _rank(path, version, prefer):
    """Lower sorts first. Prefer the target version, then shallower paths."""
    parts = path.replace(os.sep, "/").lower()
    score = 0
    if version and f"_{version}/" not in parts and f"_{version}." not in parts:
        score += 100
    for i, pat in enumerate(prefer):
        if pat in parts:
            score += i
            break
    else:
        score += len(prefer)
    return (score, path.count(os.sep), path)


def discover(root, version):
    """Locate every input the script needs. Returns a dict plus a report."""
    files = list(walk_project(root))
    found = {}
    alts = {}

    def pick(key, match, prefer=()):
        cands = [f for f in files if match(os.path.basename(f), f)]
        if not cands:
            return
        cands.sort(key=lambda p: _rank(p, version, prefer))
        found[key] = cands[0]
        if len(cands) > 1:
            alts[key] = cands[1:]

    pick("train",
         lambda b, f: b == "train.jsonl" and _looks_like_corpus(f),
         prefer=(f"dataset_{version}/", "dataset/"))
    pick("test",
         lambda b, f: b == "sft_test.jsonl",
         prefer=(f"formatted_{version}/", "formatted/"))
    pick("v8_ratings",
         lambda b, f: b == f"sft_{version}_ratings.jsonl",
         prefer=(f"llm_judge_results_{version}/", "llm_judge_results/"))

    found["evals"] = sorted(
        f for f in files
        if re.fullmatch(r"evaluation_results_v\d+\.json", os.path.basename(f))
    )
    # Judge ratings: one per model name, preferring the target version's dir.
    ratings = {}
    for f in sorted(files, key=lambda p: _rank(p, version, (f"llm_judge_results_{version}/",))):
        b = os.path.basename(f)
        if b.endswith("_ratings.jsonl"):
            ratings.setdefault(b, f)
    found["ratings"] = sorted(ratings.values())
    if len(ratings) != len([f for f in files if f.endswith("_ratings.jsonl")]):
        alts["ratings"] = [
            f for f in files
            if f.endswith("_ratings.jsonl") and f not in found["ratings"]
        ]

    gen = [os.path.dirname(f) for f in files
           if f.replace(os.sep, "/").endswith("quality/validator.py")]
    if gen:
        found["generator_root"] = os.path.dirname(sorted(gen, key=len)[0])

    return found, alts


def report_discovery(root, version, found, alts):
    print(f"\nDiscovery under {os.path.abspath(root)} (version={version})")
    for key, label in [("train", "corpus"), ("test", "test metadata"),
                       ("v8_ratings", f"{version} ratings"),
                       ("generator_root", "generator package")]:
        v = found.get(key)
        print(f"  {label:<20} {os.path.relpath(v, root) if v else '(not found)'}")
    ev = found.get("evals", [])
    print(f"  {'eval JSONs':<20} {len(ev)} found"
          + (f": {', '.join(sorted(os.path.basename(e) for e in ev))}" if ev else ""))
    rt = found.get("ratings", [])
    print(f"  {'ratings files':<20} {len(rt)} found")
    for key, others in alts.items():
        print(f"  NOTE: {len(others)} other candidate(s) for '{key}' ignored:")
        for o in others[:4]:
            print(f"        {os.path.relpath(o, root)}")
        if len(others) > 4:
            print(f"        ... and {len(others)-4} more")
    print("  Pin any of these explicitly with --train / --test / --evals if wrong.")

FAILURES = []
CHECKS = 0


def check(name, expected, actual, tol=0):
    global CHECKS
    CHECKS += 1
    if isinstance(expected, float) or isinstance(actual, float):
        ok = abs(float(expected) - float(actual)) <= tol
    else:
        ok = expected == actual
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAILURES.append(name)
    print(f"  [{status}] {name:<58} paper={expected!s:<14} actual={actual!s}")
    return ok


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── Table I: corpus composition ───────────────────────────────────────
def table_i(records):
    print("\nTable I — corpus composition")
    tier = Counter()
    level = Counter()
    risk = Counter()
    over = Counter()
    sports = set()
    cats = set()
    scens = set()
    quality = []
    for r in records:
        tier[r.get("complexity_tier")] += 1
        level[r.get("data_completeness_level")] += 1
        risk[r.get("risk_level")] += 1
        over[r.get("overreaching_classification")] += 1
        sports.add(r.get("sport"))
        cats.add(r.get("sport_category"))
        scens.add(r.get("scenario_type"))
        if r.get("quality_score") is not None:
            quality.append(r["quality_score"])

    check("total records", 40000, sum(tier.values()))
    check("distinct sports", 66, len(sports))
    check("distinct sport categories", 10, len(cats))
    check("distinct scenario types", 19, len(scens))
    check("mean quality score", 0.8585, round(statistics.mean(quality), 4), 0.0001)

    for k, v in [(1, 13805), (2, 12063), (3, 14132)]:
        check(f"tier {k}", v, tier[k])
    for k, v in [(1, 6256), (2, 15972), (3, 13895), (4, 3877)]:
        check(f"data level {k}", v, level[k])
    for k, v in [("moderate", 15829), ("moderate_to_high", 6617), ("low", 5489),
                 ("high", 4540), ("low_with_detraining_flag", 2836),
                 ("low_if_managed", 2656), ("critical", 1706), ("variable", 327)]:
        check(f"risk label {k}", v, risk[k])
    for k, v in [("normal_adaptation", 23008), ("functional_overreaching", 13570),
                 ("non_functional_overreaching", 1716),
                 ("overtraining_syndrome", 1706)]:
        check(f"overreaching label {k}", v, over[k])


# ── Table II: scenario/label determinism ──────────────────────────────
PAPER_SCENARIOS = {
    "undertraining":               (2836, "low_with_detraining_flag", "LOW",      "normal_adaptation",           1207),
    "acwr_spike":                  (2824, "high",                     "HIGH",     "functional_overreaching",     0),
    "normal_progressive":          (2767, "low",                      "LOW",      "normal_adaptation",           165),
    "taper":                       (2722, "low",                      "LOW",      "normal_adaptation",           2),
    "post_competition":            (2656, "low_if_managed",           "LOW",      "normal_adaptation",           1038),
    "fixture_congestion":          (2506, "moderate_to_high",         "MODERATE", "functional_overreaching",     0),
    "preseason_intensification":   (2425, "moderate_to_high",         "MODERATE", "functional_overreaching",     0),
    "monotony_problem":            (2396, "moderate",                 "MODERATE", "functional_overreaching",     0),
    "illness_return":              (2394, "moderate",                 "MODERATE", "normal_adaptation",           0),
    "altitude_camp":               (2342, "moderate",                 "MODERATE", "normal_adaptation",           0),
    "youth_growth_spurt":          (1767, "moderate",                 "MODERATE", "normal_adaptation",           0),
    "wellness_crash_normal_load":  (1749, "moderate",                 "MODERATE", "normal_adaptation",           0),
    "high_acwr_stable_physiology": (1733, "moderate",                 "MODERATE", "functional_overreaching",     0),
    "heat_acclimatization":        (1731, "moderate",                 "MODERATE", "normal_adaptation",           0),
    "travel_jet_lag":              (1717, "moderate",                 "MODERATE", "normal_adaptation",           0),
    "early_overreaching":          (1716, "high",                     "HIGH",     "non_functional_overreaching", 0),
    "overtraining_syndrome":       (1706, "critical",                 "CRITICAL", "overtraining_syndrome",       0),
    "double_session_accumulation": (1686, "moderate_to_high",         "MODERATE", "functional_overreaching",     0),
    "recreational_minimal_data":   (327,  "variable",                 "VARIABLE", "normal_adaptation",           0),
}


def table_ii(records):
    print("\nTable II — scenario-to-label determinism")
    n = Counter()
    risk = defaultdict(set)
    over = defaultdict(set)
    hdr = defaultdict(set)
    contra = Counter()
    for r in records:
        s = r["scenario_type"]
        n[s] += 1
        risk[s].add(r["risk_level"])
        over[s].add(r["overreaching_classification"])
        o = r.get("output_coach", "") or ""
        m = re.search(r"RISK LEVEL:\s*(\w+)", o)
        hdr[s].add(m.group(1) if m else "NONE")
        if "critically suppressed" in o and m and m.group(1) == "LOW":
            contra[s] += 1

    # the central claim: zero within-scenario variance
    multi = [s for s in n if len(risk[s]) > 1 or len(over[s]) > 1 or len(hdr[s]) > 1]
    check("scenarios with >1 risk/over/header value (must be 0)", 0, len(multi))

    for s, (cnt, rl, hh, oc, ct) in PAPER_SCENARIOS.items():
        check(f"{s}: n", cnt, n[s])
        check(f"{s}: risk label", rl, next(iter(risk[s]), None))
        check(f"{s}: header", hh, next(iter(hdr[s]), None))
        check(f"{s}: overreaching", oc, next(iter(over[s]), None))
        check(f"{s}: HRV-suppressed+LOW", ct, contra[s])

    check("total contradiction records", 2412, sum(contra.values()))


# ── Table III: contradictions by ACWR band ────────────────────────────
def table_iii(records):
    print("\nTable III — contradictions by ACWR band")
    band = Counter()
    crit_bands = Counter()
    crit_total = 0
    ots_total = 0
    for r in records:
        o = r.get("output_coach", "") or ""
        a = re.search(r"ACWR of ([\d.]+)", o)
        m = re.search(r"RISK LEVEL: (\w+)", o)
        if not (a and m):
            continue
        v = float(a.group(1))
        b = "<0.8" if v < 0.8 else "0.8-1.3" if v <= 1.3 else "1.3-1.5" if v <= 1.5 else ">1.5"
        if m.group(1) == "CRITICAL":
            crit_total += 1
            crit_bands[b] += 1
        if r["overreaching_classification"] == "overtraining_syndrome":
            ots_total += 1
        if "critically suppressed" in o and m.group(1) == "LOW":
            band[b] += 1

    for b, v in [("<0.8", 1865), ("0.8-1.3", 547), ("1.3-1.5", 0), (">1.5", 0)]:
        check(f"contradictions in ACWR {b}", v, band[b])
    check("CRITICAL records total", 1706, crit_total)
    check("CRITICAL == overtraining_syndrome count", ots_total, crit_total)
    check("CRITICAL records outside ACWR<0.8 (must be 0)", 0,
          crit_total - crit_bands["<0.8"])


# ── Table VI: revision trajectory ─────────────────────────────────────
PAPER_EVAL = {
    "sft_only_greedy": (0.691, 0.755, 0.092, 0.792, 0.974, 10, 3,  0.758, 153.1),
    "sft_v2":          (0.691, 0.675, 0.104, 0.794, 0.992, 4,  0,  0.458, 168.4),
    "sft_v2_sampled":  (0.691, 0.660, 0.129, 0.702, 0.948, 13, 13, 0.352, 165.7),
    "sft_v3":          (0.712, 0.735, 0.104, 0.842, 0.994, 3,  0,  0.518, 176.0),
    "sft_v4":          (0.716, 0.649, 0.136, 0.856, 0.956, 22, 0,  0.482, 178.8),
    "sft_v5":          (0.701, 0.720, 0.124, 0.934, 0.982, 8,  1,  0.552, 176.0),
    "sft_v6":          (0.697, 0.701, 0.136, 0.942, 0.972, 3,  11, 0.512, 171.2),
    "sft_v7":          (0.704, 0.720, 0.117, 0.948, 0.986, 3,  4,  0.554, 173.5),
    "sft_v8":          (0.701, 0.668, 0.099, 0.960, 0.994, 3,  0,  0.504, 173.2),
}

PAPER_JUDGE = {
    "sft_only_greedy": (3.00, 4.25, 5.75),
    "sft_only":        (3.24, 4.33, 5.78),
    "sft_v2":          (4.38, 4.80, 6.37),
    "sft_v3":          (4.09, 5.20, 6.36),
    "sft_v4":          (4.23, 4.99, 6.28),
    "sft_v5":          (4.23, 5.11, 6.41),
    "sft_v6":          (4.11, 5.01, 6.29),
    "sft_v7":          (3.89, 5.08, 6.32),
    "sft_v8":          (4.10, 4.90, 6.31),
}


def table_vi_eval(paths):
    print("\nTable VI — rule metrics from evaluation_results_v*.json")
    seen = {}
    ceilings = defaultdict(set)
    for p in paths:
        d = json.load(open(p))
        c = d["reference_reward_ceiling"]
        for name, m in d["models"].items():
            seen[name] = (c, m)
            ceilings[name].add(c)
    for name, exp in PAPER_EVAL.items():
        if name not in seen:
            print(f"  [SKIP] {name} — not found in supplied eval files")
            continue
        c, m = seen[name]
        ce, rw, sd, ra, w1, sv, uk, ov, wd = exp
        check(f"{name}: ceiling", ce, c, 0.0005)
        check(f"{name}: reward", rw, m["reward_mean"], 0.0005)
        check(f"{name}: reward sd", sd, m["reward_std"], 0.0005)
        check(f"{name}: risk acc", ra, m["risk_accuracy"], 0.0005)
        check(f"{name}: within-1", w1, m["risk_within_one"], 0.0005)
        check(f"{name}: severe", sv, m["risk_severe_errors"])
        check(f"{name}: unknown", uk, m["risk_unknown"])
        check(f"{name}: overreaching", ov, m["overreaching_accuracy"], 0.0005)
        check(f"{name}: words", wd, m["word_count_mean"], 0.05)

    print("\n  Shared-test-set groups (same ceiling => same test split):")
    groups = defaultdict(list)
    for name, (c, _) in seen.items():
        groups[c].append(name)
    for c, names in sorted(groups.items()):
        marker = "  <-- COMPARABLE" if len(names) > 1 else ""
        print(f"    ceiling {c}: {', '.join(sorted(names))}{marker}")
    print("  NOTE: rows sharing a ceiling are like-for-like. Confirm the paper's")
    print("  per-row version labels and its 'not comparable' caveat match this.")


def table_vi_judge_files(paths):
    print("\nTable VI/VIII — judge means from *_ratings.jsonl")
    for path in sorted(paths, key=os.path.basename):
        name = os.path.basename(path).replace("_ratings.jsonl", "")
        ca, ac, cl = [], [], []
        for r in load_jsonl(path):
            if "error" in r or "clinical_accuracy" not in r:
                continue
            ca.append(r["clinical_accuracy"])
            ac.append(r["actionability"])
            cl.append(r["clarity"])
        if not ca:
            continue
        check(f"{name}: n ratings", 438, len(ca))
        if name in PAPER_JUDGE:
            c, a, l = PAPER_JUDGE[name]
            check(f"{name}: clinical", c, round(statistics.mean(ca), 2), 0.005)
            check(f"{name}: actionability", a, round(statistics.mean(ac), 2), 0.005)
            check(f"{name}: clarity", l, round(statistics.mean(cl), 2), 0.005)


# ── Table VIII: v8 stratified judge (needs the test metadata) ─────────
PAPER_STRATA = {
    "high_acwr_stable_physiology": (2.00, 3.78, 5.67, 9),
    "preseason_intensification":   (2.67, 4.76, 6.05, 21),
    "undertraining":               (2.90, 3.59, 5.77, 39),
    "monotony_problem":            (3.25, 4.88, 6.04, 24),
    "travel_jet_lag":              (3.27, 3.47, 5.93, 15),
    "double_session_accumulation": (3.33, 4.20, 5.87, 15),
    "acwr_spike":                  (3.47, 5.83, 6.72, 36),
    "normal_progressive":          (3.71, 3.38, 5.71, 21),
    "heat_acclimatization":        (3.75, 5.33, 6.33, 24),
    "post_competition":            (3.78, 4.00, 6.06, 18),
    "fixture_congestion":          (3.79, 5.12, 6.30, 33),
    "taper":                       (4.26, 4.30, 6.22, 27),
    "recreational_minimal_data":   (4.67, 4.33, 6.33, 3),
    "illness_return":              (4.71, 5.21, 6.46, 24),
    "youth_growth_spurt":          (5.24, 5.52, 6.71, 21),
    "early_overreaching":          (5.29, 5.24, 6.57, 21),
    "altitude_camp":               (5.41, 5.19, 6.67, 27),
    "overtraining_syndrome":       (5.60, 6.36, 6.79, 42),
    "wellness_crash_normal_load":  (5.72, 5.72, 6.78, 18),
}


def table_viii(ratings_path, test_path):
    print("\nTable VIII — v8 stratified judge scores")
    meta = {}
    for i, r in enumerate(load_jsonl(test_path)):
        meta[i] = r.get("scenario_type", "?")
    groups = defaultdict(lambda: ([], [], []))
    for r in load_jsonl(ratings_path):
        if "error" in r or "clinical_accuracy" not in r:
            continue
        s = meta.get(r["id"])
        if s is None:
            continue
        g = groups[s]
        g[0].append(r["clinical_accuracy"])
        g[1].append(r["actionability"])
        g[2].append(r["clarity"])
    for s, (c, a, l, n) in PAPER_STRATA.items():
        if s not in groups:
            print(f"  [SKIP] {s} — no ratings joined")
            continue
        g = groups[s]
        check(f"{s}: n", n, len(g[0]))
        check(f"{s}: clinical", c, round(statistics.mean(g[0]), 2), 0.005)
        check(f"{s}: actionability", a, round(statistics.mean(g[1]), 2), 0.005)
        check(f"{s}: clarity", l, round(statistics.mean(g[2]), 2), 0.005)


# ── Table IX: reward decomposition ────────────────────────────────────
def composite_components(t):
    text = (t or "").lower()
    c = {}
    if any(f"risk level: {r}" in text or f"your status: {r}" in text
           for r in ["low", "moderate", "high", "critical"]):
        c["risk"] = 0.30
    elif any(r in text for r in ["low", "moderate", "high", "critical"]):
        c["risk"] = 0.15
    else:
        c["risk"] = 0.0
    if re.search(r"\d+\s*[–\-]\s*\d+\s*%", text):
        c["rec"] = 0.25
    elif re.search(r"rpe\s*[<≤]\s*\d", text):
        c["rec"] = 0.22
    elif re.search(r"\d+\s*(day|week|session)", text):
        c["rec"] = 0.18
    elif re.search(r"\d+%", text):
        c["rec"] = 0.15
    elif any(v in text for v in ["reduce", "maintain", "rest", "monitor", "suspend"]):
        c["rec"] = 0.10
    else:
        c["rec"] = 0.0
    c["over"] = 0.20 if any(x in text for x in [
        "non-functional overreaching", "functional overreaching",
        "overtraining syndrome", "normal adaptation", "undertraining"]) else 0.0
    c["esc"] = 0.15 if any(x in text for x in [
        "physician", "medical review", "doctor", "clinical assessment",
        "escalate", "seek support"]) else 0.0
    w = len((t or "").split())
    c["len"] = 0.10 if 200 <= w <= 450 else (0.05 if 100 <= w <= 600 else 0.0)
    return c, w


PAPER_REWARD = {
    "athlete":          (0.299, 0.180, 0.028, 0.105, 0.056, 0.669, 156.2),
    "coach":            (0.299, 0.186, 0.126, 0.079, 0.057, 0.746, 176.9),
    "sports_scientist": (0.150, 0.187, 0.175, 0.106, 0.069, 0.688, 195.8),
}


def table_ix(records):
    print("\nTable IX — reward decomposition")
    acc = defaultdict(list)
    words = defaultdict(list)
    totals = defaultdict(list)
    headers = defaultdict(Counter)
    for r in records:
        for aud in ["athlete", "coach", "sports_scientist"]:
            t = r.get("output_" + aud, "")
            c, w = composite_components(t)
            for k, v in c.items():
                acc[(aud, k)].append(v)
            words[aud].append(w)
            totals[aud].append(min(1.0, round(sum(c.values()), 3)))
            m = re.match(r"\s*([A-Z][A-Z /&()-]{2,}):", t or "")
            headers[aud][m.group(1) if m else "NONE"] += 1

    for aud, exp in PAPER_REWARD.items():
        for i, k in enumerate(["risk", "rec", "over", "esc", "len"]):
            check(f"{aud}: {k}", exp[i],
                  round(statistics.mean(acc[(aud, k)]), 3), 0.0005)
        check(f"{aud}: total", exp[5],
              round(statistics.mean(totals[aud]), 3), 0.0005)
        check(f"{aud}: mean words", exp[6],
              round(statistics.mean(words[aud]), 1), 0.05)

    pooled = [x for v in totals.values() for x in v]
    check("pooled ceiling (must equal reported 0.701)", 0.701,
          round(statistics.mean(pooled), 3), 0.0005)

    for aud, hh in [("athlete", "YOUR STATUS"), ("coach", "RISK LEVEL"),
                    ("sports_scientist", "OVERALL RISK CLASSIFICATION")]:
        check(f"{aud}: all briefs open '{hh}'", 40000, headers[aud][hh])


# ── Table IV: schema validator (actually measure it) ──────────────────
def _renest(r):
    """Rebuild the pre-export nested shape DatasetValidator expects.

    The HuggingFace exporter flattens `metadata` and `ground_truth_labels` into
    top-level columns, so the validator cannot be run against the released
    corpus directly -- every record fails with missing_metadata_*. This inverts
    that flattening so the validator sees the shape it was written for.
    """
    if "metadata" in r and "ground_truth_labels" in r:
        return r
    return {
        **r,
        "ground_truth_labels": {
            "acwr_value": r.get("acwr_value"),
            "acwr_zone": r.get("acwr_zone", ""),
            "risk_level": r.get("risk_level", ""),
            "overreaching_classification": r.get("overreaching_classification", ""),
            "complexity_tier": r.get("complexity_tier", 1),
            "conflicting_signals": r.get("conflicting_signals", False),
        },
        "metadata": {
            "scenario_type": r.get("scenario_type", ""),
            "sport": r.get("sport", ""),
            "sport_category": r.get("sport_category", ""),
            "athlete_level": r.get("athlete_level", ""),
            "training_phase": r.get("training_phase", ""),
            "data_completeness_level": r.get("data_completeness_level", 2),
            "complexity_tier": r.get("complexity_tier", 1),
            "source": r.get("source", "synthetic"),
        },
    }


def check_validator(records, generator_root=None):
    print("\nTable IV — schema validator, measured")
    try:
        for cand in filter(None, [generator_root, "loadbrief_generator", "."]):
            if cand not in sys.path:
                sys.path.insert(0, cand)
        from quality.validator import DatasetValidator
    except Exception as e:
        print(f"  [SKIP] could not import DatasetValidator: {e}")
        print("  Pass --root pointing at the project, or check that")
        print("  loadbrief_generator/quality/validator.py exists.")
        return

    flat = "metadata" not in (records[0] if records else {})
    if flat:
        print("  Released corpus is in flattened export form; reconstructing the")
        print("  nested shape the validator expects before running.")

    v = DatasetValidator()
    flagged = 0
    kinds = Counter()
    for r in records:
        res = v.validate(_renest(r))
        if not res["passed"]:
            flagged += 1
            for f in res["failures"]:
                kinds[f] += 1
    print(f"  validator flags {flagged} / {len(records)} records")
    for k, n in kinds.most_common(10):
        print(f"    {n:6d}  {k}")
    check("schema validator flags on released corpus", 0, flagged)
    print("  NOTE: the validator's cross-field check is one-directional -- it")
    print("  verifies that high/critical labels appear in the brief text and")
    print("  applies no check to benign labels. A 0 here is consistent with the")
    print("  paper's structural claim; it does not independently confirm it.")


def main():
    ap = argparse.ArgumentParser(
        description="Verify the LoadBrief paper's tables against source data.")
    ap.add_argument("--root", default=".",
                    help="LoadBrief project root; searched recursively (default: .)")
    ap.add_argument("--version", default="v8",
                    help="corpus/model revision to verify (default: v8)")
    ap.add_argument("--train", default=None, help="override: corpus train.jsonl")
    ap.add_argument("--evals", nargs="*", default=None,
                    help="override: evaluation_results_v*.json files")
    ap.add_argument("--ratings", default=None,
                    help="override: directory of *_ratings.jsonl")
    ap.add_argument("--v8-ratings", dest="v8_ratings", default=None,
                    help="override: ratings file for the target version")
    ap.add_argument("--test", default=None,
                    help="override: formatted sft_test.jsonl for the Table VIII join")
    ap.add_argument("--check-validator", action="store_true",
                    help="measure the schema-validator row of Table IV")
    a = ap.parse_args()

    print("=" * 90)
    print("  LoadBrief paper table verification")
    print("=" * 90)

    if not os.path.isdir(a.root):
        sys.exit(f"--root {a.root!r} is not a directory")

    found, alts = discover(a.root, a.version)
    report_discovery(a.root, a.version, found, alts)

    # Explicit flags win over discovery.
    train = a.train or found.get("train")
    test = a.test or found.get("test")
    v8_ratings = a.v8_ratings or found.get("v8_ratings")
    evals = a.evals if a.evals is not None else found.get("evals", [])
    if a.ratings:
        ratings_files = sorted(
            os.path.join(a.ratings, f) for f in os.listdir(a.ratings)
            if f.endswith("_ratings.jsonl"))
    else:
        ratings_files = found.get("ratings", [])

    if not train:
        sys.exit(
            f"\nNo corpus found under {a.root!r}. Expected a train.jsonl inside a\n"
            f"dataset directory (e.g. dataset_{a.version}/train.jsonl).\n"
            "Pass --train explicitly if it lives elsewhere.")

    records = list(load_jsonl(train))
    print(f"\nLoaded {len(records):,} records from "
          f"{os.path.relpath(train, a.root)}")

    table_i(records)
    table_ii(records)
    table_iii(records)
    table_ix(records)
    if a.check_validator:
        check_validator(records, found.get("generator_root"))
    if evals:
        table_vi_eval(evals)
    else:
        print("\nTable VI rule metrics — SKIPPED (no evaluation_results_v*.json found)")
    if ratings_files:
        table_vi_judge_files(ratings_files)
    else:
        print("\nTable VI judge means — SKIPPED (no *_ratings.jsonl found)")
    if v8_ratings and test:
        table_viii(v8_ratings, test)
    else:
        missing = []
        if not v8_ratings:
            missing.append(f"sft_{a.version}_ratings.jsonl")
        if not test:
            missing.append("formatted sft_test.jsonl")
        print(f"\nTable VIII — SKIPPED (not found: {', '.join(missing)})")

    print("\n" + "=" * 90)
    if FAILURES:
        print(f"  {len(FAILURES)} of {CHECKS} checks FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
    else:
        print(f"  All {CHECKS} checks passed.")
    print("=" * 90)
    print("\nNot covered by this script:")
    print("  - Table V (reachability): run reachability_audit.py against each")
    print("    surviving dataset_v*/train.jsonl, or mark the table historical.")
    print("  - Table IV consistency-checker row: run consistency_check.py.")
    print("  - Table IV label-agreement row: run label_agreement.py.")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
