#!/usr/bin/env python3
"""overreaching_selftest.py — why did overreaching accuracy collapse after v1?

HYPOTHESIS
----------
`evaluate_all.py` scores overreaching by string-matching clinical vocabulary
inside a section it locates by ALL-CAPS header. That measurement is only as good
as the corpus's phrasing. If the generator's wording changed between corpus
revisions, the extractor's hit rate changes with it -- and reported accuracy
falls even if the model learned the task exactly as well.

THE TEST
--------
Feed each corpus's OWN ground-truth briefs through the real extractor and score
them against their own labels. A ground-truth brief is by definition the correct
answer, so a perfect extractor scores 1.00 on every corpus. Whatever it actually
scores is the ceiling the extractor imposes on any model evaluated against that
corpus.

Read the SELF column as: "if the model reproduced its training data perfectly,
this is the overreaching accuracy it would be reported as achieving."

    SELF ~ 1.00 across revisions  -> extraction is sound; the collapse is real,
                                     look at the label mapping instead.
    SELF drops at v2              -> the collapse is a measurement artifact and
                                     the reported numbers are not comparable.

Usage:
    python3 overreaching_selftest.py --root .
    python3 overreaching_selftest.py --corpora dataset/train.jsonl \\
        "Compiled Datasets/dataset_v2/train.jsonl"
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
SKIP_PREFIX = ("sft_checkpoint", "grpo_checkpoint", "final_model", "baseline_outputs")

AUD_FIELD = {
    "athlete": "output_athlete",
    "coach": "output_coach",
    "sports_scientist": "output_sports_scientist",
}


def walk(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(SKIP_PREFIX)]
        for fn in fns:
            yield os.path.join(dp, fn)


def find_corpora(root):
    out = []
    for f in walk(root):
        if os.path.basename(f) != "train.jsonl":
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        if "input_narrative" in r and "scenario_type" in r:
                            out.append(f)
                        break
        except Exception:
            pass
    def key(p):
        m = re.search(r"dataset_v(\d+)", p)
        return (int(m.group(1)) if m else 0, p)
    return sorted(out, key=key)


def label_of(path):
    m = re.search(r"dataset_v(\d+)", path)
    return f"v{m.group(1)}" if m else "v1?"


def load_extractor(root):
    """Use the real extractor from evaluate_all.py, not a reimplementation."""
    cands = [f for f in walk(root) if os.path.basename(f) == "evaluate_all.py"]
    if not cands:
        sys.exit("evaluate_all.py not found; pass --root at the project directory")
    d = os.path.dirname(sorted(cands, key=lambda p: p.count(os.sep))[0]) or "."
    if d not in sys.path:
        sys.path.insert(0, d)
    import evaluate_all as E
    return E, os.path.join(d, "evaluate_all.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--corpora", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="rows per corpus (default: all)")
    a = ap.parse_args()

    E, epath = load_extractor(a.root)
    print(f"extractor: {epath}\n")

    corpora = a.corpora or find_corpora(a.root)
    if not corpora:
        sys.exit("no corpora found")

    print("SELF-EXTRACTION ACCURACY ON GROUND-TRUTH BRIEFS")
    print("(a perfect extractor scores 1.00; anything less is a ceiling on any")
    print(" model evaluated against that corpus)\n")
    hdr = f"{'corpus':10s}{'n':>7}" + "".join(f"{a_:>13}" for a_ in AUD_FIELD) + f"{'pooled':>9}{'unknown':>9}"
    print(hdr)
    print("-" * len(hdr))

    detail = {}
    for path in corpora:
        lab = label_of(path)
        ok = Counter()
        tot = Counter()
        unk = 0
        confusion = defaultdict(Counter)
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n += 1
                gt = E._norm_over(r.get("overreaching_classification"))
                for aud, field in AUD_FIELD.items():
                    brief = r.get(field, "")
                    if not brief:
                        continue
                    tot[aud] += 1
                    pred = E._norm_over(E.extract_predicted_overreaching(brief))
                    if pred == gt:
                        ok[aud] += 1
                    if pred == "unknown":
                        unk += 1
                    confusion[gt][pred] += 1
                if a.limit and n >= a.limit:
                    break
        cells = "".join(f"{ok[x]/max(tot[x],1):>13.3f}" for x in AUD_FIELD)
        pooled = sum(ok.values()) / max(sum(tot.values()), 1)
        print(f"{lab:10s}{n:>7}{cells}{pooled:>9.3f}{unk/max(sum(tot.values()),1):>9.1%}")
        detail[lab] = confusion

    print("\nCONFUSION ON GROUND TRUTH (label -> what the extractor read)")
    for lab, conf in detail.items():
        rows = []
        for gt, preds in sorted(conf.items()):
            wrong = {k: v for k, v in preds.items() if k != gt}
            if wrong:
                tot = sum(preds.values())
                top = ", ".join(f"{k}:{v}" for k, v in
                                sorted(wrong.items(), key=lambda x: -x[1])[:3])
                rows.append(f"    {gt:30s} {preds[gt]:>7}/{tot:<7} misread as {top}")
        if rows:
            print(f"\n  {lab}:")
            for r in rows:
                print(r)
        else:
            print(f"\n  {lab}: clean -- every label extracted correctly")

    print("\nHOW TO READ THIS")
    print("  If SELF is high everywhere, extraction is sound and the overreaching")
    print("  collapse is a real change in what the corpus teaches -- compare the")
    print("  scenario-to-class mappings next.")
    print("  If SELF drops between revisions, the reported accuracies are measured")
    print("  through different lenses and are not comparable across those rows.")


if __name__ == "__main__":
    main()
