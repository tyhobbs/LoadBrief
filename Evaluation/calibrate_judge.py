#!/usr/bin/env python3
"""Reference-brief judge calibration.

Runs the EXACT same Gemini judge (same prompt, same call) over the generator's
own ground-truth briefs for a scenario. These briefs are correct by construction
— they are the ideal outputs. If the judge scores them LOW with the same
complaints it gave the model's briefs, the low score is a judge ceiling, not a
model defect: the judge disagrees with correct sports science on this scenario.

Reuses llm_judge.py's JUDGE_PROMPT and call_gemini so the measurement is
identical to the real evaluation — nothing is reconstructed.

Usage:
  export GEMINI_API_KEY=...
  python3 calibrate_judge.py --scenario overtraining_syndrome --n 8
  python3 calibrate_judge.py --scenario overtraining_syndrome --n 8 --audience coach
"""
import json, argparse, sys

# Import the real judge machinery — do NOT reimplement it.
import llm_judge as J

# Default judge model strings. Kept HERE (not hardcoded in llm_judge.py) so a
# model deprecation is a flag change, not a code edit. Verify against your
# provider console before a large run — these move often.
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
DEFAULT_GPT_MODEL = "gpt-4o"


def call_claude_model(prompt, model):
    """Claude judge with a configurable model string.

    Reasoning models return a ThinkingBlock before the TextBlock, so
    content[0] is NOT the answer — scan for the text block. max_tokens is
    raised because thinking consumes the same budget.
    """
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return "".join(getattr(b, "text", "") for b in resp.content)


def call_gpt_model(prompt, model):
    """OpenAI judge with a configurable model string."""
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

AUD_FIELD = {
    "athlete": "output_athlete",
    "coach": "output_coach",
    "sports_scientist": "output_sports_scientist",
}


def build_prompt_input(rec):
    """Reconstruct the INPUT the judge sees, matching run_baselines' prompt.

    The judge prompt has {prompt} (monitoring input) and {completion} (brief).
    For reference briefs the 'completion' is the ground-truth output; the
    'prompt' is the same input_narrative the model was given.
    """
    return rec.get("input_narrative", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v2/train.jsonl",
                    help="source of ground-truth briefs")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--audience", default="coach",
                    choices=list(AUD_FIELD))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--judge", default="gemini",
                    choices=["gemini", "claude", "gpt"],
                    help="judge backend")
    ap.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    ap.add_argument("--gpt-model", default=DEFAULT_GPT_MODEL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    field = AUD_FIELD[args.audience]

    if args.judge == "gemini":
        judge_fn = J.call_gemini
        judge_label = getattr(J, "GEMINI_MODEL", "gemini")
    elif args.judge == "claude":
        judge_fn = lambda p: call_claude_model(p, args.claude_model)
        judge_label = args.claude_model
    else:
        judge_fn = lambda p: call_gpt_model(p, args.gpt_model)
        judge_label = args.gpt_model

    # Pull the first N ground-truth briefs for this scenario
    refs = []
    for line in open(args.data):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("scenario_type") == args.scenario:
            brief = r.get(field, "")
            if brief and len(brief.strip()) > 50:
                refs.append(r)
                if len(refs) >= args.n:
                    break

    if not refs:
        sys.exit(f"No {args.scenario} briefs with a {field} field in {args.data}")

    print("=" * 70)
    print(f"  JUDGE CALIBRATION on GROUND-TRUTH briefs")
    print(f"  scenario={args.scenario}  audience={args.audience}  n={len(refs)}")
    print(f"  judge={judge_label} (same prompt as real eval)")
    print("=" * 70)
    print("  These briefs are correct by construction. Low scores here mean")
    print("  the JUDGE is the ceiling, not the model.\n")

    ratings = []
    for i, r in enumerate(refs):
        prompt = J.JUDGE_PROMPT.format(
            prompt=build_prompt_input(r),
            completion=r[field],
        )
        try:
            raw = judge_fn(prompt)
        except Exception as e:
            print(f"  [{i}] judge error: {e}")
            continue
        try:
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            rat = json.loads(clean)
        except Exception:
            print(f"  [{i}] unparseable: {raw[:80]}")
            continue
        rat["id"] = i
        ratings.append(rat)
        print(f"  [clin={rat.get('clinical_accuracy')} "
              f"act={rat.get('actionability')} "
              f"clar={rat.get('clarity')}]  {rat.get('brief_comment','')[:150]}")

    if ratings:
        import statistics
        cl = [r["clinical_accuracy"] for r in ratings if "clinical_accuracy" in r]
        from collections import Counter
        print("\n" + "=" * 70)
        print(f"  Ground-truth {args.scenario} clinical accuracy: "
              f"mean {statistics.mean(cl):.2f}  dist {dict(sorted(Counter(cl).items()))}")
        print("=" * 70)
        print("  Compare to your MODEL's score for this scenario. If they match,")
        print("  the model reached the judge's ceiling — the residual is not a")
        print("  fixable defect but a limit of the judge on this scenario.")

    if args.out:
        with open(args.out, "w") as f:
            for r in ratings:
                f.write(json.dumps(r) + "\n")
        print(f"\n  saved: {args.out}")


if __name__ == "__main__":
    main()