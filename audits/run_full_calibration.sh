#!/usr/bin/env bash
# run_full_calibration.sh — calibrate the judge on ground-truth briefs across
# every scenario and register.
#
#   chmod +x run_full_calibration.sh
#   ./run_full_calibration.sh                 # gemini, all 19 scenarios, 3 registers
#   ./run_full_calibration.sh claude          # same with the claude judge
#   N=12 ./run_full_calibration.sh            # more briefs per cell (tighter CIs)
#   AUDIENCES=coach ./run_full_calibration.sh # coach register only (1/3 the cost)
#
# Resumable: a cell whose output file already exists and is non-empty is
# skipped, so you can interrupt with Ctrl-C and rerun. Delete a file to redo it.

set -u

JUDGE="${1:-gemini}"
N="${N:-10}"
DATA="${DATA:-Compiled Datasets/dataset_v8/train.jsonl}"
OUTDIR="${OUTDIR:-calibration}"
AUDIENCES="${AUDIENCES:-athlete coach sports_scientist}"
SLEEP="${SLEEP:-1}"          # seconds between cells; raise if rate-limited

SCENARIOS="acwr_spike altitude_camp double_session_accumulation early_overreaching
fixture_congestion heat_acclimatization high_acwr_stable_physiology illness_return
monotony_problem normal_progressive overtraining_syndrome post_competition
preseason_intensification recreational_minimal_data taper travel_jet_lag
undertraining wellness_crash_normal_load youth_growth_spurt"

export PYTHONPATH="${PYTHONPATH:-.}"
mkdir -p "$OUTDIR"

total=0; done_=0; skipped=0; failed=0
for s in $SCENARIOS; do for a in $AUDIENCES; do total=$((total+1)); done; done
echo "judge=$JUDGE  n=$N  cells=$total  data=$DATA"
echo "output -> $OUTDIR/"
echo

i=0
for s in $SCENARIOS; do
  for a in $AUDIENCES; do
    i=$((i+1))
    out="$OUTDIR/${s}__${a}__${JUDGE}.jsonl"
    if [ -s "$out" ]; then
      echo "[$i/$total] skip (exists)  $s / $a"
      skipped=$((skipped+1))
      continue
    fi
    echo "[$i/$total] $s / $a"
    if python3 calibrate_judge.py \
         --data "$DATA" \
         --scenario "$s" \
         --audience "$a" \
         --n "$N" \
         --judge "$JUDGE" \
         --out "$out" > "$OUTDIR/${s}__${a}__${JUDGE}.log" 2>&1
    then
      if [ -s "$out" ]; then
        done_=$((done_+1))
      else
        echo "      -> no ratings parsed; see $OUTDIR/${s}__${a}__${JUDGE}.log"
        rm -f "$out"
        failed=$((failed+1))
      fi
    else
      echo "      -> FAILED; see $OUTDIR/${s}__${a}__${JUDGE}.log"
      rm -f "$out"
      failed=$((failed+1))
    fi
    sleep "$SLEEP"
  done
done

echo
echo "completed=$done_  skipped=$skipped  failed=$failed  of $total"
echo "Rerun this script to retry failures (completed cells are skipped)."
echo "Then: python3 summarize_calibration.py --calibration $OUTDIR"
