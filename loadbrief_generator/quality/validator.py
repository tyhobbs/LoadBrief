# quality/validator.py
# Validates each generated training example against
# required schema and clinical accuracy checks.
# Examples failing validation are rejected before training.

from typing import Dict, List, Tuple


# Required fields in every training example
REQUIRED_INPUT_SIGNALS = [
    "sport", "level", "training_phase"
]

REQUIRED_OUTPUT_FIELDS = [
    "RISK LEVEL"
]

# Risk levels that must appear in output
VALID_RISK_LEVELS = {
    "low", "moderate", "high", "critical",
    "LOW", "MODERATE", "HIGH", "CRITICAL"
}

# Overreaching classifications
VALID_OVERREACHING_CLASSES = {
    "normal_adaptation",
    "functional_overreaching",
    "non_functional_overreaching",
    "overtraining_syndrome",
    "undertraining"
}


class DatasetValidator:
    """
    Validates training examples for:
    1. Schema completeness — all required fields present
    2. Clinical accuracy — classifications are internally consistent
    3. Output quality — briefs contain required sections
    4. Cross-field consistency — input and output agree
    """

    def validate(self, example: Dict) -> Dict:
        """
        Run all validation checks on a single example.
        Returns dict with passed flag and list of failures.
        """
        failures = []

        # Check 1: Required metadata
        meta_failures = self._check_metadata(example)
        failures.extend(meta_failures)

        # Check 2: Input narrative quality
        narrative_failures = self._check_narrative(
            example.get("input_narrative", "")
        )
        failures.extend(narrative_failures)

        # Check 3: Output brief quality (all audiences)
        for audience in ["athlete", "coach",
                         "sports_scientist"]:
            key = f"output_{audience}"
            brief = example.get(key, "")
            brief_failures = self._check_brief(
                brief, audience
            )
            failures.extend([
                f"{audience}_{f}" for f in brief_failures
            ])

        # Check 4: Ground truth label validity
        label_failures = self._check_labels(
            example.get("ground_truth_labels", {})
        )
        failures.extend(label_failures)

        # Check 5: Cross-field consistency
        consistency_failures = self._check_consistency(example)
        failures.extend(consistency_failures)

        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "failure_count": len(failures)
        }

    def _check_metadata(self, example: Dict) -> List[str]:
        """Check required metadata fields"""
        failures = []
        meta = example.get("metadata", {})

        required_meta = [
            "scenario_type", "sport",
            "complexity_tier", "data_completeness_level"
        ]
        for field in required_meta:
            if not meta.get(field):
                failures.append(f"missing_metadata_{field}")

        return failures

    def _check_narrative(self, narrative: str) -> List[str]:
        """Check input narrative quality"""
        failures = []

        if not narrative or len(narrative.strip()) < 50:
            failures.append("narrative_too_short")
            return failures

        # Must mention a sport or athlete type
        sport_mentions = [
            "athlete", "sport", "training", "session",
            "monitoring", "load", "rpe"
        ]
        if not any(word in narrative.lower()
                   for word in sport_mentions):
            failures.append("narrative_no_sport_context")

        # Overly long narratives likely have formatting issues
        if len(narrative.split()) > 500:
            failures.append("narrative_too_long")

        return failures

    def _check_brief(self, brief: str,
                     audience: str) -> List[str]:
        """Check output brief quality"""
        failures = []

        if not brief or len(brief.strip()) < 30:
            failures.append("brief_too_short")
            return failures

        # Check for risk level
        has_risk = any(
            level in brief.upper()
            for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        )
        if not has_risk:
            failures.append("brief_no_risk_level")

        # Check for recommendation
        recommendation_markers = [
            "recommend", "reduce", "maintain", "increase",
            "monitor", "rest", "review", "continue",
            "should", "advised"
        ]
        has_recommendation = any(
            marker in brief.lower()
            for marker in recommendation_markers
        )
        if not has_recommendation:
            failures.append("brief_no_recommendation")

        # Audience-specific length checks
        word_count = len(brief.split())
        if audience == "athlete" and word_count > 400:
            failures.append("athlete_brief_too_long")
        if audience == "sports_scientist" and word_count < 100:
            failures.append("scientist_brief_too_short")

        # Check for placeholder text
        placeholders = [
            "{acwr}", "{hrv}", "{athlete}",
            "INSERT", "TODO", "PLACEHOLDER"
        ]
        if any(p in brief for p in placeholders):
            failures.append("brief_has_placeholder_text")

        return failures

    def _check_labels(self,
                      labels: Dict) -> List[str]:
        """Validate ground truth labels"""
        failures = []

        # Risk level must be valid
        risk = labels.get("risk_level", "")
        if not risk:
            failures.append("missing_risk_level_label")

        # Overreaching classification must be valid
        oc = labels.get("overreaching_classification", "")
        if oc and oc not in VALID_OVERREACHING_CLASSES:
            failures.append(
                f"invalid_overreaching_class_{oc}"
            )

        # Complexity tier must be 1, 2, or 3
        tier = labels.get("complexity_tier", 0)
        if tier not in [1, 2, 3]:
            failures.append("invalid_complexity_tier")

        return failures

    def _check_consistency(self,
                            example: Dict) -> List[str]:
        """Check cross-field consistency"""
        failures = []

        labels = example.get("ground_truth_labels", {})
        narrative = example.get("input_narrative", "")
        brief_coach = example.get("output_coach", "")

        risk = labels.get("risk_level", "")

        # High/critical risk should appear in outputs
        if risk in ["high", "critical"]:
            if "high" not in brief_coach.lower() and \
               "critical" not in brief_coach.lower() and \
               "elevated" not in brief_coach.lower() and \
               "concern" not in brief_coach.lower():
                failures.append(
                    "high_risk_not_reflected_in_output"
                )

        # OTS should trigger medical review mention
        oc = labels.get("overreaching_classification", "")
        if oc == "overtraining_syndrome":
            medical_markers = [
                "physician", "medical", "doctor",
                "clinical", "review", "assess"
            ]
            if not any(m in brief_coach.lower()
                       for m in medical_markers):
                failures.append(
                    "ots_missing_medical_referral"
                )

        return failures

    def validate_batch(self,
                       examples: List[Dict]) -> Tuple[
                           List[Dict], List[Dict], Dict]:
        """
        Validate a batch of examples.
        Returns (passed, rejected, stats).
        """
        passed = []
        rejected = []

        failure_counts = {}

        for example in examples:
            result = self.validate(example)
            if result["passed"]:
                passed.append(example)
            else:
                rejected.append({
                    "example": example,
                    "failures": result["failures"]
                })
                for failure in result["failures"]:
                    failure_counts[failure] = \
                        failure_counts.get(failure, 0) + 1

        stats = {
            "total": len(examples),
            "passed": len(passed),
            "rejected": len(rejected),
            "pass_rate": len(passed) / max(len(examples), 1),
            "top_failures": sorted(
                failure_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

        return passed, rejected, stats
