# llm_judge.py
# Use an LLM to rate briefs from each model variant — a scalable stand-in
# for expert human review.
#
# Judges (pick one):
#   --judge gemini   Google Gemini 2.5 Flash, FREE tier (recommended)
#                    ~15 RPM / 1,500 requests-per-day, no credit card.
#                    Different model family from Llama, so it avoids the
#                    self-preference bias a Llama-based judge would have.
#                    NOTE: Google may use FREE-tier prompts for training.
#                    Fine for these synthetic, non-sensitive briefs; do NOT
#                    use the free tier for real patient/clinical data.
#   --judge claude   Anthropic Claude (paid, ~$30 for 500 briefs)
#   --judge gpt4     OpenAI GPT-4o (paid)
#
# Setup — pick the matching key:
#   export GEMINI_API_KEY=...        (from https://aistudio.google.com, no card)
#   export ANTHROPIC_API_KEY=sk-ant-...
#   export OPENAI_API_KEY=sk-...
#
# Install the matching SDK:
#   pip install google-genai          # for gemini
#   pip install anthropic             # for claude
#   pip install openai                # for gpt4
#
# Usage:
#   python3 llm_judge.py --judge gemini --baseline all --n 100
#   python3 llm_judge.py --judge gemini --baseline main
#
# Free-tier budgeting: the daily cap is ~1,500 requests. 11 models x 100
# briefs = 1,100 calls (fits in one day). Judging all 11 at n=500 would be
# 5,500 calls — over the daily cap, so either keep n=100 or judge a subset
# (e.g. zero_shot, sft_only, main, ablation_none) for the comparisons that
# matter most.

import os
import json
import time
import argparse
import statistics
from pathlib import Path

BASELINE_DIR = "./baseline_outputs"
OUTPUT_DIR   = "./llm_judge_results"


class QuotaExhausted(Exception):
    """Raised when the daily free-tier quota is hit, to stop the run cleanly
    instead of writing hundreds of error rows that would poison the files."""
    pass

JUDGE_PROMPT = """You are an expert sports scientist evaluating an AI-generated load management brief for an athlete.

INPUT (monitoring data given to the AI):
{prompt}

AI-GENERATED BRIEF:
{completion}

Rate this brief on three dimensions using a 1-7 scale.

1. CLINICAL ACCURACY: Does the brief reflect sound sports science principles?
   (Acute:Chronic Workload Ratio interpretation, HRV trends, overreaching classification, etc.)
   1 = dangerously wrong, 4 = adequate, 7 = excellent clinical reasoning

2. ACTIONABILITY: Could a coach apply these recommendations directly tomorrow?
   1 = vague platitudes, 4 = somewhat usable, 7 = specific quantitative actions

3. CLARITY: Is the brief well-organized and easy to understand?
   1 = incoherent, 4 = readable, 7 = professionally written

Respond in this exact JSON format:
{{
  "clinical_accuracy": <integer 1-7>,
  "actionability": <integer 1-7>,
  "clarity": <integer 1-7>,
  "brief_comment": "<one sentence justifying the lowest score>"
}}

Output only the JSON object. No other text."""


# ── Judge backends ────────────────────────────────────────────────────

def call_claude(prompt):
    """Call Claude API for one rating."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_gpt4(prompt):
    """Call OpenAI API for one rating."""
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# Gemini free tier is ~15 requests/minute. Throttle to ~13 RPM so we stay
# safely under the cap without external rate-limit errors.
GEMINI_MODEL        = "gemini-2.5-flash"
GEMINI_MIN_INTERVAL = 4.5          # seconds between calls (~13 RPM)
_LAST_GEMINI_CALL   = [0.0]        # mutable holder for last-call timestamp
_GEMINI_CLIENT      = [None]       # cache the client across calls


def call_gemini(prompt):
    """Call Google Gemini 2.5 Flash (free tier) for one rating.

    Self-throttles to respect the free-tier rate limit and retries with
    backoff on quota/rate errors. Requests JSON output and disables the
    model's 'thinking' budget so the short rating isn't crowded out.
    """
    from google import genai
    from google.genai import types

    # Throttle to stay under the free-tier RPM cap
    elapsed = time.time() - _LAST_GEMINI_CALL[0]
    if elapsed < GEMINI_MIN_INTERVAL:
        time.sleep(GEMINI_MIN_INTERVAL - elapsed)

    # Build (and cache) the client
    if _GEMINI_CLIENT[0] is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        _GEMINI_CLIENT[0] = genai.Client(api_key=api_key)
    client = _GEMINI_CLIENT[0]

    # Generation config: deterministic, JSON output, no thinking budget.
    cfg_kwargs = dict(
        max_output_tokens=512,
        temperature=0.0,
        response_mime_type="application/json",
    )
    try:
        # Supported on 2.5 Flash; guarded in case the SDK version differs.
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass
    config = types.GenerateContentConfig(**cfg_kwargs)

    last_err = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config,
            )
            _LAST_GEMINI_CALL[0] = time.time()
            return resp.text or ""
        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # Daily free-tier quota exhausted: no point continuing the run.
            # 'RESOURCE_EXHAUSTED' with a per-day limit will not clear by
            # waiting a few seconds, so stop the whole run cleanly rather
            # than writing hundreds of error rows.
            if "resource_exhausted" in msg or "exceeded your current quota" in msg \
                    or "perday" in msg or "per day" in msg:
                raise QuotaExhausted(str(e))

            # Transient errors worth retrying with backoff:
            #   429 / rate            -> per-minute rate limit
            #   nodename / servname / getaddrinfo / temporary failure -> DNS
            #   connection / disconnect / timed out / timeout         -> network
            #   503 / unavailable / server                            -> server blip
            transient = any(k in msg for k in [
                "429", "rate",
                "nodename", "servname", "getaddrinfo", "temporary failure",
                "connection", "disconnect", "timed out", "timeout",
                "503", "unavailable", "server",
            ])
            if transient and attempt < 4:
                # exponential backoff: 5, 10, 20, 40s
                time.sleep(5 * (2 ** attempt))
                continue
            raise
    if last_err:
        raise last_err
    return ""


# ── Parsing and scoring ───────────────────────────────────────────────

def parse_rating(text):
    """Extract JSON ratings from LLM response."""
    text = (text or "").strip()
    # Remove code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object inside the text
        import re
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None


def judge_baseline(baseline_name, judge_fn, output_dir, n_examples=100):
    """Judge one baseline's outputs."""
    input_path = f"{BASELINE_DIR}/{baseline_name}.jsonl"
    if not Path(input_path).exists():
        print(f"  Skipping {baseline_name} — no output file")
        return None

    Path(output_dir).mkdir(exist_ok=True)
    output_path = f"{output_dir}/{baseline_name}_ratings.jsonl"

    # Load completions to judge
    examples = []
    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n_examples:
                break
            examples.append(json.loads(line.strip()))

    # Resume from existing ratings — but ONLY count rows that succeeded.
    # Error rows are dropped so they get retried rather than skipped forever.
    good_rows = {}
    if Path(output_path).exists():
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r and "clinical_accuracy" in r:
                        good_rows[r["id"]] = r
    rated_ids = set(good_rows.keys())

    # Rewrite the file containing only the good rows, so prior error rows
    # are cleared and we can cleanly append fresh attempts.
    with open(output_path, "w") as f:
        for r in good_rows.values():
            f.write(json.dumps(r) + "\n")

    print(f"\nJudging {baseline_name}: {len(examples)} total, "
          f"{len(rated_ids)} already rated (errors will be retried)")

    t0 = time.time()
    done = 0
    hit_quota = False
    with open(output_path, "a") as f:
        for i, ex in enumerate(examples):
            if ex["id"] in rated_ids:
                continue

            prompt = JUDGE_PROMPT.format(
                prompt=ex["prompt"][:1500],          # truncate huge prompts
                completion=ex["completion"][:2000],   # truncate huge completions
            )

            try:
                response = judge_fn(prompt)
                rating   = parse_rating(response)
                if rating is None:
                    rating = {"error": "parse_failed", "raw": (response or "")[:200]}
            except QuotaExhausted as e:
                # Daily quota gone — stop now, do NOT write an error row.
                print(f"\n  Daily quota exhausted while judging {baseline_name}.")
                print(f"  Stopping cleanly. {done} new ratings saved this run.")
                print(f"  Re-run after the quota resets; it will resume here.")
                hit_quota = True
                break
            except Exception as e:
                rating = {"error": str(e)}

            record = {"id": ex["id"], "baseline": baseline_name, **rating}
            f.write(json.dumps(record) + "\n")
            f.flush()
            done += 1

            if done % 10 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done} new | {rate*60:.0f}/min")

    # Compute summary
    all_ratings = []
    with open(output_path, "r") as f:
        for line in f:
            r = json.loads(line.strip())
            if "error" not in r and "clinical_accuracy" in r:
                all_ratings.append(r)

    if not all_ratings:
        return None, hit_quota

    summary = {
        "baseline":              baseline_name,
        "n_rated":               len(all_ratings),
        "clinical_accuracy_mean": round(statistics.mean(r["clinical_accuracy"] for r in all_ratings), 2),
        "actionability_mean":     round(statistics.mean(r["actionability"] for r in all_ratings), 2),
        "clarity_mean":           round(statistics.mean(r["clarity"] for r in all_ratings), 2),
        "overall_mean":           round(statistics.mean(
            (r["clinical_accuracy"] + r["actionability"] + r["clarity"]) / 3
            for r in all_ratings
        ), 2),
    }
    return summary, hit_quota


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge",    type=str,
                        choices=["gemini", "claude", "gpt4"], default="gemini")
    parser.add_argument("--baseline", type=str, default="all")
    parser.add_argument("--n",        type=int, default=100,
                        help="Examples per baseline (default 100)")
    args = parser.parse_args()

    if args.judge == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set")
            return
        judge_fn = call_claude
    elif args.judge == "gpt4":
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set")
            return
        judge_fn = call_gpt4
    else:  # gemini
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            print("ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
            print("Get a free key at https://aistudio.google.com (no credit card).")
            return
        judge_fn = call_gemini

    print("=" * 60)
    print(f"  LLM-as-judge — {args.judge}")
    print("=" * 60)

    # Find baselines to judge
    if args.baseline == "all":
        baselines = [p.stem for p in Path(BASELINE_DIR).glob("*.jsonl")]
    else:
        baselines = [args.baseline]

    # Merge with any existing summary so a partial/looped run doesn't wipe
    # models judged earlier.
    summaries = {}
    summary_path = f"{OUTPUT_DIR}/summary.json"
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    if Path(summary_path).exists():
        try:
            with open(summary_path) as f:
                summaries = json.load(f)
        except Exception:
            summaries = {}

    for b in baselines:
        summary, hit_quota = judge_baseline(b, judge_fn, OUTPUT_DIR, args.n)
        if summary:
            summaries[b] = summary
            # Save after every model so progress survives an interruption.
            with open(summary_path, "w") as f:
                json.dump(summaries, f, indent=2)
        if hit_quota:
            print("\nStopping the run — daily quota reached. "
                  "Re-run after reset to finish the remaining models.")
            break

    # Print comparison
    print("\n" + "=" * 80)
    print(f"{'Baseline':<25} {'Clinical':<10} {'Action':<10} {'Clarity':<10} {'Overall':<10}")
    print("=" * 80)
    for name, s in sorted(summaries.items(), key=lambda x: -x[1]["overall_mean"]):
        print(f"{name:<25} {s['clinical_accuracy_mean']:<10} "
              f"{s['actionability_mean']:<10} {s['clarity_mean']:<10} "
              f"{s['overall_mean']:<10}")
    print("=" * 80)

    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
