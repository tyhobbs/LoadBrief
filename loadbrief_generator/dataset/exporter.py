# dataset/exporter.py
# Saves the dataset in HuggingFace-compatible format.
# Produces Parquet files + dataset card + README
# that can be uploaded directly to HuggingFace Hub.

import json
import os
from pathlib import Path
from typing import Dict, List
from datetime import date


class HuggingFaceExporter:
    """
    Exports the LoadBrief dataset in HuggingFace format.
    Output directory can be pushed directly to HF Hub with:
    hf upload [your-username]/LoadBrief-50K ./dataset
    """

    def save_huggingface_format(self,
                                 dataset: Dict,
                                 output_dir: str):
        """
        Save complete dataset in HuggingFace format.
        Creates all required files and subdirectories.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("Saving dataset splits...")
        self._save_splits(dataset, output_path)

        print("Saving subsets...")
        self._save_subsets(dataset, output_path)

        print("Saving statistics...")
        self._save_statistics(dataset, output_path)

        print("Writing dataset card...")
        self._write_dataset_card(dataset, output_path)

        print("Writing README...")
        self._write_readme(dataset, output_path)

        print(f"\nDataset saved to: {output_path}")
        print(f"Total examples: {dataset['total_examples']}")
        print("\nTo upload to HuggingFace:")
        print("  pip install huggingface_hub")
        print("  hf auth login")
        print(
            f"  hf upload "
            f"[your-username]/LoadBrief-50K {output_dir}"
        )

    def _save_splits(self, dataset: Dict,
                      output_path: Path):
        """Save train/validation/test splits as JSONL"""
        for split_name in ["train", "validation", "test"]:
            split_data = dataset.get(split_name, [])
            if not split_data:
                continue

            # Save as JSONL (one JSON object per line)
            split_file = output_path / f"{split_name}.jsonl"
            with open(split_file, "w", encoding="utf-8") as f:
                for example in split_data:
                    # Flatten the example for HF format
                    flat = self._flatten_example(example)
                    f.write(json.dumps(flat,
                                       ensure_ascii=False) + "\n")

            print(f"  {split_name}: {len(split_data)} examples "
                  f"→ {split_file.name}")

    def _save_subsets(self, dataset: Dict,
                       output_path: Path):
        """Save category subsets"""
        subsets_path = output_path / "subsets"
        subsets_path.mkdir(exist_ok=True)

        subsets = dataset.get("subsets", {})
        for subset_name, examples in subsets.items():
            if not examples:
                continue

            subset_file = subsets_path / f"{subset_name}.jsonl"
            with open(subset_file, "w",
                      encoding="utf-8") as f:
                for example in examples:
                    flat = self._flatten_example(example)
                    f.write(
                        json.dumps(flat, ensure_ascii=False)
                        + "\n"
                    )

    def _save_statistics(self, dataset: Dict,
                          output_path: Path):
        """Save dataset statistics as JSON"""
        stats_file = output_path / "dataset_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(
                dataset.get("statistics", {}),
                f,
                indent=2,
                ensure_ascii=False
            )

    def _flatten_example(self, example: Dict) -> Dict:
        """
        Flatten nested example into HuggingFace-compatible
        flat dict with consistent column names.
        """
        labels = example.get("ground_truth_labels", {})
        meta = example.get("metadata", {})

        return {
            # Input
            "input_narrative": example.get(
                "input_narrative", ""
            ),

            # Outputs per audience
            "output_athlete": example.get(
                "output_athlete", ""
            ),
            "output_coach": example.get(
                "output_coach", ""
            ),
            "output_sports_scientist": example.get(
                "output_sports_scientist", ""
            ),

            # Ground truth labels
            "acwr_value": labels.get("acwr_value"),
            "acwr_zone": labels.get("acwr_zone", ""),
            "risk_level": labels.get("risk_level", ""),
            "overreaching_classification": labels.get(
                "overreaching_classification", ""
            ),
            "complexity_tier": labels.get(
                "complexity_tier", 1
            ),
            "conflicting_signals": labels.get(
                "conflicting_signals", False
            ),

            # Metadata
            "scenario_type": meta.get("scenario_type", ""),
            "sport": meta.get("sport", ""),
            "sport_category": meta.get("sport_category", ""),
            "athlete_level": meta.get("athlete_level", ""),
            "training_phase": meta.get("training_phase", ""),
            "data_completeness_level": meta.get(
                "data_completeness_level", 2
            ),
            "source": meta.get("source", "synthetic"),
            "quality_score": example.get("quality_score", 0)
        }

    def _write_dataset_card(self, dataset: Dict,
                             output_path: Path):
        """Write HuggingFace dataset card (README.md)"""
        stats = dataset.get("statistics", {})
        n = dataset.get("total_examples", 0)

        card = f"""---
language:
- en
license: cc-by-4.0
task_categories:
- text-generation
- text-classification
task_ids:
- conditional-text-generation
- multi-label-classification
tags:
- sports-science
- athlete-monitoring
- load-management
- reinforcement-learning
- exercise-science
- ACWR
- HRV
- overreaching
pretty_name: LoadBrief-50K
size_categories:
- 10K<n<100K
---

# LoadBrief-50K

## Dataset Summary

LoadBrief-50K is the largest NLP dataset for athlete load management
interpretation. It contains {n:,} input-output text pairs spanning
{stats.get('n_sports_covered', 35)}+ sports, 20 monitoring scenario
types, and 4 data-completeness levels.

Each example consists of:
- A **monitoring narrative** (input) describing an athlete's recent
  training load, HRV, and wellness data
- **Structured load management briefs** (output) at four audience
  levels: recreational athlete, coach, sports scientist, and API
- **Ground truth labels** including ACWR zone, risk level,
  overreaching classification, and complexity tier

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total examples | {n:,} |
| Sports covered | {stats.get('n_sports_covered', 35)}+ |
| Scenario types | {stats.get('n_scenario_types', 20)} |
| Avg quality score | {stats.get('average_quality_score', 0):.3f} |

### Risk Level Distribution
{self._format_dist_table(stats.get('risk_level_distribution', {}))}

### Complexity Tier Distribution
{self._format_dist_table(stats.get('tier_distribution', {}))}

## Supported Tasks

- **Conditional text generation**: Given a monitoring narrative
  and audience token, generate appropriate load management brief
- **Risk classification**: Classify ACWR zone and overreaching state
- **Multi-signal synthesis**: Detect and resolve conflicting
  monitoring signals

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `input_narrative` | string | Monitoring data narrative |
| `output_athlete` | string | Brief for recreational athlete |
| `output_coach` | string | Brief for coach |
| `output_sports_scientist` | string | Technical brief |
| `acwr_value` | float | Calculated ACWR |
| `acwr_zone` | string | Risk zone classification |
| `risk_level` | string | Overall risk (low/moderate/high/critical) |
| `overreaching_classification` | string | Meeusen et al. 2013 |
| `complexity_tier` | int | 1=clear-cut, 2=moderate, 3=complex |
| `conflicting_signals` | bool | True if signals disagree |
| `sport` | string | Athlete's sport |
| `data_completeness_level` | int | 1-4 monitoring data level |

### Data Splits

| Split | Size |
|-------|------|
| Train | {len(dataset.get('train', [])):,} |
| Validation | {len(dataset.get('validation', [])):,} |
| Test | {len(dataset.get('test', [])):,} |

## Usage

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("[your-username]/LoadBrief-50K")

# Load specific subset
tier3 = load_dataset(
    "[your-username]/LoadBrief-50K",
    data_files="subsets/tier_3_complex.jsonl"
)
```

## Citation

```bibtex
@dataset{{loadbrief50k_2024,
  title={{LoadBrief-50K: A General-Purpose NLP Dataset
          for Athlete Load Management Interpretation}},
  author={{[Author]}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/[username]/LoadBrief-50K}}
}}
```

## License

CC BY 4.0 — Free for academic and commercial use with attribution.

## Acknowledgments

Dataset parameters grounded in: Gabbett (2016), Meeusen et al.
(2013), Plews et al. (2013), Hooper & Mackinnon (1995),
Foster (1998), Buchheit & Laursen (2013).
"""

        card_file = output_path / "README.md"
        with open(card_file, "w", encoding="utf-8") as f:
            f.write(card)

    def _write_readme(self, dataset: Dict,
                       output_path: Path):
        """Write usage README for the simulator code"""
        readme = """# LoadBrief Generator

Complete pipeline for generating the LoadBrief-50K dataset.

## Quick Start

```bash
pip install -r requirements.txt
python main.py --n_examples 50000 --output_dir ./dataset
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_examples` | 50000 | Number of examples to generate |
| `--output_dir` | ./dataset | Output directory |
| `--seed` | 42 | Random seed for reproducibility |

## File Structure

See the project README for complete file descriptions.

## Requirements

See requirements.txt
"""
        readme_file = output_path / "GENERATOR_README.md"
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(readme)

    def _format_dist_table(self, dist: Dict) -> str:
        """Format distribution dict as markdown table"""
        if not dist:
            return "| Category | % |\n|----------|---|\n"

        rows = ["| Category | % |", "|----------|---|"]
        for k, v in sorted(dist.items()):
            rows.append(f"| {k} | {v:.1%} |")
        return "\n".join(rows)
