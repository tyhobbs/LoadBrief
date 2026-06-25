# main_parallel.py
# Parallel dataset generator optimized for M4 Max/Ultra
# python3 main_parallel.py --n_examples 50000 --output_dir ./dataset

import os
import sys
import json
import random
import argparse
import multiprocessing as mp
from pathlib import Path

# Resolve paths at module load time — before any subprocess is spawned
# These become module-level constants that are pickled into subprocesses
_ROOT      = os.path.dirname(os.path.abspath(__file__))
_GENERATOR = os.path.join(_ROOT, "loadbrief_generator")

# Add to path for the main process
if _GENERATOR not in sys.path:
    sys.path.insert(0, _GENERATOR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _setup_subprocess_path():
    """
    Call at the top of every function that runs in a subprocess.
    Spawned processes on macOS start with a clean sys.path so
    we must re-add the generator path explicitly every time.
    """
    if _GENERATOR not in sys.path:
        sys.path.insert(0, _GENERATOR)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)


def generate_single_example(scenario_name: str, seed: int) -> dict:
    """Generate one complete training example. Runs in a subprocess."""
    _setup_subprocess_path()

    import random as rnd
    import numpy as np
    rnd.seed(seed)
    np.random.seed(seed)

    from config import (
        SPORT_CATEGORIES, TRAINING_PHASES,
        HRV_THRESHOLDS, ACWR_THRESHOLDS,
        WELLNESS_NORMS, DATA_LEVELS, AUDIENCE_PROFILES
    )
    from athlete_generator import AthleteProfileGenerator
    from simulator.scenarios import ALL_SCENARIOS
    from simulator.time_series import TimeSeriesSimulator
    from simulator.metrics import (
        calculate_acwr, classify_acwr_zone,
        analyze_hrv_trend, analyze_wellness_trend,
        calculate_monotony, calculate_strain,
        classify_overreaching_state
    )
    from simulator.data_levels import DataLevelFilter
    from narrative.narrator import MonitoringNarrator
    from brief_generator.signal_synthesizer import SignalSynthesizer
    from brief_generator.template_mixer import TemplateMixer

    scenario = ALL_SCENARIOS[scenario_name]

    athlete_gen  = AthleteProfileGenerator(SPORT_CATEGORIES, TRAINING_PHASES, HRV_THRESHOLDS)
    ts_sim       = TimeSeriesSimulator(ACWR_THRESHOLDS, HRV_THRESHOLDS, WELLNESS_NORMS)
    data_filter  = DataLevelFilter(DATA_LEVELS)
    narrator     = MonitoringNarrator()
    synthesizer  = SignalSynthesizer({'acwr': ACWR_THRESHOLDS, 'hrv': HRV_THRESHOLDS})
    mixer        = TemplateMixer(AUDIENCE_PROFILES)

    athlete = athlete_gen.generate()
    ts      = ts_sim.simulate(athlete, scenario, weeks=4)

    loads = [d['session_load'] for d in ts]
    acwr  = calculate_acwr(loads)
    acwr['zone']     = classify_acwr_zone(acwr.get('acwr', 0), athlete['sport_category'], ACWR_THRESHOLDS)
    acwr['monotony'] = calculate_monotony(loads[-7:])
    acwr['strain']   = calculate_strain(loads)

    hrv_vals      = [d['hrv'] for d in ts if d.get('hrv') is not None]
    hrv           = analyze_hrv_trend(hrv_vals, athlete['baseline_hrv'], HRV_THRESHOLDS)
    wellness_hist = [d['wellness'] for d in ts if d.get('wellness')]
    wellness      = analyze_wellness_trend(wellness_hist)

    data_level = scenario.special_parameters.get(
        'data_level',
        rnd.choices([1, 2, 3, 4], weights=[15, 40, 35, 10])[0]
    )
    filtered  = data_filter.apply(acwr, hrv, wellness, data_level)
    synthesis = synthesizer.synthesize(filtered['acwr'], filtered['hrv'], filtered['wellness'], scenario, athlete)
    narrative = narrator.generate(athlete, ts, filtered, data_level, scenario)

    briefs = {}
    for audience in ['athlete', 'coach', 'sports_scientist']:
        briefs[f'output_{audience}'] = mixer.generate_brief(
            synthesis, filtered['acwr'], filtered['hrv'],
            filtered['wellness'], athlete, audience, scenario
        )

    return {
        'input_narrative': narrative,
        **briefs,
        'ground_truth_labels': {
            'acwr_value':                acwr.get('acwr'),
            'acwr_zone':                 acwr.get('zone', ''),
            'risk_level':                scenario.risk_level,
            'overreaching_classification': scenario.overreaching_class,
            'complexity_tier':           scenario.complexity_tier,
            'conflicting_signals':       scenario.conflicting_signals,
            'signal_conflicts':          scenario.signal_conflicts
        },
        'metadata': {
            'scenario_type':            scenario_name,
            'sport':                    athlete['sport'],
            'sport_category':           athlete['sport_category'],
            'athlete_level':            athlete['level'],
            'training_phase':           athlete['phase'],
            'data_completeness_level':  data_level,
            'complexity_tier':          scenario.complexity_tier,
            'source':                   'synthetic_simulator_v1'
        }
    }


def worker_fn(task_queue, result_queue, worker_id: int):
    """Worker process — pulls tasks, generates examples, returns results."""
    _setup_subprocess_path()

    from quality.validator import DatasetValidator
    from quality.filter import QualityFilter

    validator      = DatasetValidator()
    quality_filter = QualityFilter()

    while True:
        task = task_queue.get()
        if task is None:  # poison pill — shut down
            break

        scenario_name, seed = task
        try:
            example      = generate_single_example(scenario_name, seed)
            validation   = validator.validate(example)
            quality_score = quality_filter.score(example)

            if validation['passed'] and quality_filter.passes(example):
                example['quality_score'] = quality_score
                result_queue.put(('ok', example))
            else:
                result_queue.put(('rejected', None))

        except Exception as e:
            result_queue.put(('error', str(e)))


def build_scenario_pool():
    """Build weighted scenario pool — Tier 1:40%, Tier 2:35%, Tier 3:25%"""
    _setup_subprocess_path()
    from simulator.scenarios import TIER_1_SCENARIOS, TIER_2_SCENARIOS, TIER_3_SCENARIOS
    return (
        TIER_1_SCENARIOS * 8 +
        TIER_2_SCENARIOS * 7 +
        TIER_3_SCENARIOS * 5
    )


def save_dataset(examples: list, output_dir: str):
    """Assemble and save dataset in HuggingFace format."""
    _setup_subprocess_path()
    from dataset.builder import DatasetBuilder
    from dataset.exporter import HuggingFaceExporter

    builder  = DatasetBuilder()
    dataset  = builder.build(examples)
    exporter = HuggingFaceExporter()
    exporter.save_huggingface_format(dataset, output_dir)
    return dataset['statistics']


def main():
    parser = argparse.ArgumentParser(description="Generate LoadBrief dataset")
    parser.add_argument('--n_examples',  type=int,  default=50000)
    parser.add_argument('--output_dir',  type=str,  default='./dataset')
    parser.add_argument('--seed',        type=int,  default=42)
    parser.add_argument('--n_workers',   type=int,  default=None)
    args = parser.parse_args()

    n_workers = args.n_workers or max(1, mp.cpu_count() - 2)

    print(f"LoadBrief Parallel Generator")
    print(f"Target examples : {args.n_examples:,}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers         : {n_workers} (of {mp.cpu_count()} cores)")
    print(f"Seed            : {args.seed}")
    print(f"Generator path  : {_GENERATOR}")
    print()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    scenario_pool = build_scenario_pool()
    total_tasks   = int(args.n_examples * 1.15)

    tasks = [
        (random.choice(scenario_pool), args.seed + i)
        for i in range(total_tasks)
    ]

    task_queue   = mp.Queue()
    result_queue = mp.Queue()

    # Start workers
    workers = []
    for worker_id in range(n_workers):
        p = mp.Process(target=worker_fn, args=(task_queue, result_queue, worker_id))
        p.start()
        workers.append(p)

    # Feed tasks
    for task in tasks:
        task_queue.put(task)
    for _ in range(n_workers):
        task_queue.put(None)  # poison pills

    # Collect results
    import time
    examples   = []
    rejected   = 0
    errors     = 0
    processed  = 0
    first_errors = []
    start_time = time.time()

    while processed < total_tasks:
        status, payload = result_queue.get()
        processed += 1

        if status == 'ok':
            examples.append(payload)
        elif status == 'rejected':
            rejected += 1
        else:
            errors += 1
            if len(first_errors) < 3:
                first_errors.append(payload)

        if processed % 500 == 0 or processed == total_tasks:
            elapsed   = time.time() - start_time
            rate      = processed / max(elapsed, 0.001)
            remaining = (total_tasks - processed) / max(rate, 0.001)
            print(
                f"Progress: {len(examples):,}/{args.n_examples:,} clean "
                f"| {rejected:,} rejected | {errors:,} errors "
                f"| {rate*60:.0f}/min "
                f"| ~{remaining/60:.1f}min remaining",
                end='\r'
            )

        if len(examples) >= args.n_examples:
            break

    # Stop workers
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    elapsed  = time.time() - start_time
    examples = examples[:args.n_examples]

    print(f"\n\nGeneration complete:")
    print(f"  Clean examples : {len(examples):,}")
    print(f"  Rejected       : {rejected:,}")
    print(f"  Errors         : {errors:,}")
    print(f"  Acceptance rate: {len(examples)/max(processed,1):.1%}")
    print(f"  Time elapsed   : {elapsed/60:.1f} minutes")
    print(f"  Rate           : {processed/max(elapsed,1)*60:.0f} examples/min")

    # Show errors if any
    if first_errors:
        print(f"\nFirst error messages:")
        for i, err in enumerate(first_errors, 1):
            print(f"  [{i}] {err}")

    # Tier distribution
    tiers = {}
    for ex in examples:
        t = ex.get('metadata', {}).get('complexity_tier', 0)
        tiers[t] = tiers.get(t, 0) + 1
    print(f"\nTier distribution:")
    labels = {1: 'Clear-cut', 2: 'Moderate', 3: 'Complex'}
    for tier, count in sorted(tiers.items()):
        print(f"  Tier {tier} ({labels.get(tier,'?')}): {count:,} ({count/len(examples):.1%})")

    print(f"\nSaving dataset to {args.output_dir}...")
    stats = save_dataset(examples, args.output_dir)
    print(f"\nDataset saved successfully.")
    print(f"Sports covered   : {stats.get('n_sports_covered', '?')}")
    print(f"Avg quality score: {stats.get('average_quality_score', 0):.3f}")
    print(f"\nNext step: python3 format_dataset.py")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
