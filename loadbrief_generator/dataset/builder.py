# dataset/builder.py
# Assembles validated examples into a structured dataset
# with train/validation/test splits and organized subsets.

import random
from typing import List, Dict, Tuple


class DatasetBuilder:
    """
    Assembles the final LoadBrief dataset from validated examples.
    Creates train/val/test splits and category subsets.
    """

    def __init__(self, train_ratio: float = 0.80,
                 val_ratio: float = 0.10,
                 test_ratio: float = 0.10,
                 seed: int = 42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def build(self, examples: List[Dict]) -> Dict:
        """
        Build complete dataset from list of validated examples.
        Returns structured dataset dict.
        """
        random.seed(self.seed)
        random.shuffle(examples)

        # Create splits
        n = len(examples)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train = examples[:n_train]
        val = examples[n_train:n_train + n_val]
        test = examples[n_train + n_val:]

        # Create category subsets
        subsets = self._create_subsets(examples)

        # Compute dataset statistics
        stats = self._compute_statistics(examples)

        return {
            "train": train,
            "validation": val,
            "test": test,
            "subsets": subsets,
            "statistics": stats,
            "total_examples": n
        }

    def _create_subsets(self,
                         examples: List[Dict]) -> Dict:
        """
        Create organized subsets for targeted fine-tuning.
        Researchers can load only the subset they need.
        """
        subsets = {
            # By complexity tier
            "tier_1_clear_cut": [],
            "tier_2_moderate": [],
            "tier_3_complex": [],

            # By sport category
            "team_field_sports": [],
            "endurance_sports": [],
            "combat_sports": [],
            "general_population": [],
            "youth_athletes": [],
            "military_tactical": [],

            # By scenario type
            "overreaching_scenarios": [],
            "taper_scenarios": [],
            "return_scenarios": [],
            "environmental_scenarios": [],

            # By data completeness
            "level_1_minimal": [],
            "level_2_moderate": [],
            "level_3_comprehensive": [],
            "level_4_research": [],

            # Real data only (for validation)
            "real_data_only": [],

            # High quality only (quality score > 0.85)
            "high_quality": []
        }

        for ex in examples:
            meta = ex.get("metadata", {})
            labels = ex.get("ground_truth_labels", {})
            quality = ex.get("quality_score", 0)

            tier = labels.get("complexity_tier", 1)
            sport_cat = meta.get(
                "sport_category", "general_population"
            )
            scenario = meta.get("scenario_type", "")
            data_level = meta.get("data_completeness_level", 2)
            source = meta.get("source", "synthetic")

            # Tier subsets
            tier_key = f"tier_{tier}_" + {
                1: "clear_cut",
                2: "moderate",
                3: "complex"
            }.get(tier, "clear_cut")
            if tier_key in subsets:
                subsets[tier_key].append(ex)

            # Sport category subsets
            if sport_cat in subsets:
                subsets[sport_cat].append(ex)

            # Scenario type subsets
            if any(s in scenario for s in [
                "overreaching", "overtraining", "spike"
            ]):
                subsets["overreaching_scenarios"].append(ex)

            if "taper" in scenario or "competition" in scenario:
                subsets["taper_scenarios"].append(ex)

            if "return" in scenario or "illness" in scenario:
                subsets["return_scenarios"].append(ex)

            if any(s in scenario for s in [
                "altitude", "heat", "travel", "jet_lag"
            ]):
                subsets["environmental_scenarios"].append(ex)

            # Data level subsets
            level_key = f"level_{data_level}_" + {
                1: "minimal",
                2: "moderate",
                3: "comprehensive",
                4: "research"
            }.get(data_level, "moderate")
            if level_key in subsets:
                subsets[level_key].append(ex)

            # Real data subset
            if "real" in source or "literature" in source:
                subsets["real_data_only"].append(ex)

            # High quality subset
            if quality >= 0.85:
                subsets["high_quality"].append(ex)

        return subsets

    def _compute_statistics(self,
                             examples: List[Dict]) -> Dict:
        """Compute dataset statistics for the dataset card"""
        n = len(examples)
        if n == 0:
            return {}

        # Tier distribution
        tiers = [
            ex.get("ground_truth_labels", {}).get(
                "complexity_tier", 0
            )
            for ex in examples
        ]
        tier_dist = {
            f"tier_{t}": tiers.count(t) / n
            for t in [1, 2, 3]
        }

        # Risk level distribution
        risks = [
            ex.get("ground_truth_labels", {}).get(
                "risk_level", "unknown"
            )
            for ex in examples
        ]
        risk_levels = ["low", "moderate", "high", "critical"]
        risk_dist = {}
        for risk in risk_levels:
            count = sum(
                1 for r in risks
                if risk in r.lower()
            )
            risk_dist[risk] = round(count / n, 3)

        # Sport distribution (top 10)
        sports = [
            ex.get("metadata", {}).get("sport", "unknown")
            for ex in examples
        ]
        sport_counts = {}
        for s in sports:
            sport_counts[s] = sport_counts.get(s, 0) + 1
        top_sports = sorted(
            sport_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Data level distribution
        levels = [
            ex.get("metadata", {}).get(
                "data_completeness_level", 0
            )
            for ex in examples
        ]
        level_dist = {
            f"level_{l}": levels.count(l) / n
            for l in [1, 2, 3, 4]
        }

        # Overreaching classification distribution
        oc_classes = [
            ex.get("ground_truth_labels", {}).get(
                "overreaching_classification", "unknown"
            )
            for ex in examples
        ]
        oc_dist = {}
        for oc in set(oc_classes):
            oc_dist[oc] = round(oc_classes.count(oc) / n, 3)

        # Quality score statistics
        quality_scores = [
            ex.get("quality_score", 0)
            for ex in examples
        ]
        avg_quality = sum(quality_scores) / max(len(quality_scores), 1)

        return {
            "total_examples": n,
            "tier_distribution": tier_dist,
            "risk_level_distribution": risk_dist,
            "top_10_sports": dict(top_sports),
            "data_level_distribution": level_dist,
            "overreaching_distribution": oc_dist,
            "average_quality_score": round(avg_quality, 3),
            "n_sports_covered": len(sport_counts),
            "n_scenario_types": len(set(
                ex.get("metadata", {}).get(
                    "scenario_type", ""
                )
                for ex in examples
            ))
        }
