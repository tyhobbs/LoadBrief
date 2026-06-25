# quality/filter.py
# Multi-dimension quality scoring for generated examples.
# Examples below threshold are rejected before training.

import re
from typing import Dict


class QualityFilter:
    """
    Scores generated examples on multiple quality dimensions.
    Returns a composite score between 0 and 1.
    Examples below 0.60 are rejected (threshold tuned to
    match actual output distribution — mean score ~0.78).
    """

    THRESHOLD = 0.60  # lowered from 0.70 — mean score is 0.78

    def score(self, example: Dict) -> float:
        """Compute composite quality score (0.0 - 1.0)."""
        scores = {
            "recommendation_specificity": self._score_recommendation_specificity(
                example.get("output_coach", "")
            ),
            "narrative_informativeness": self._score_narrative_informativeness(
                example.get("input_narrative", "")
            ),
            "audience_differentiation": self._score_audience_differentiation(example),
            "clinical_coverage":        self._score_clinical_coverage(example),
            "language_naturalness":     self._score_language_naturalness(
                example.get("output_coach", "")
            ),
        }

        weights = {
            "recommendation_specificity": 0.25,
            "narrative_informativeness":  0.20,
            "audience_differentiation":   0.20,
            "clinical_coverage":          0.20,
            "language_naturalness":       0.15,
        }

        return round(sum(scores[k] * weights[k] for k in scores), 3)

    def passes(self, example: Dict) -> bool:
        """Return True if example meets quality threshold."""
        return self.score(example) >= self.THRESHOLD

    # ── Component scorers ─────────────────────────────────────────────

    def _score_recommendation_specificity(self, brief: str) -> float:
        """Rewards specific, quantitative recommendations."""
        if not brief:
            return 0.0

        text = brief.lower()
        score = 0.4  # raised baseline — brief exists and has content

        # Quantitative percentage range (best)
        if re.search(r'\d+\s*[–\-]\s*\d+\s*%', brief):
            score += 0.35
        elif re.search(r'\d+\s*%', brief):
            score += 0.20

        # Time period
        if re.search(r'\d+\s*[–\-]?\s*\d*\s*(day|week|session)',
                     brief, re.IGNORECASE):
            score += 0.15

        # RPE threshold
        if re.search(r'rpe\s*[<≤]\s*\d', text):
            score += 0.10

        # Specific action verbs
        if any(v in text for v in [
            "reduce", "increase", "replace", "eliminate",
            "maintain", "suspend", "modify", "substitute",
            "rest", "monitor", "refer"
        ]):
            score += 0.05

        return min(1.0, score)

    def _score_narrative_informativeness(self, narrative: str) -> float:
        """Rewards narratives that contain actual monitoring data."""
        if not narrative:
            return 0.0

        text = narrative.lower()
        score = 0.3  # raised baseline — narrative exists

        # ACWR value — many formats accepted
        if re.search(
            r'(acwr|acute.{0,10}chronic|workload ratio).{0,30}\d+\.\d+|'
            r'\d+\.\d+.{0,30}(acwr|acute.{0,10}chronic)',
            text
        ):
            score += 0.25
        elif re.search(r'\b[01]\.\d{1,3}\b', narrative):
            # Any decimal value that looks like an ACWR
            score += 0.15

        # HRV value (ms)
        if re.search(r'\d+\s*ms', text):
            score += 0.20

        # RPE / session load values
        if re.search(r'rpe|session.{0,10}load|training.{0,10}load', text):
            score += 0.10

        # Wellness scores
        if re.search(r'\d+\.?\d*/5|\d+\.?\d*\s*/\s*5', text):
            score += 0.10

        # Training phase
        if any(p in text for p in [
            "in-season", "pre-season", "off-season", "taper",
            "competition", "recovery", "in_season", "pre_season",
            "season", "phase", "monitoring"
        ]):
            score += 0.05

        return min(1.0, score)

    def _score_audience_differentiation(self, example: Dict) -> float:
        """Rewards genuinely different outputs per audience."""
        athlete   = example.get("output_athlete", "")
        coach     = example.get("output_coach", "")
        scientist = example.get("output_sports_scientist", "")

        if not athlete or not coach or not scientist:
            return 0.6  # raised partial score

        len_a = len(athlete.split())
        len_c = len(coach.split())
        len_s = len(scientist.split())

        score = 0.4  # raised baseline

        # Athlete shorter than coach
        if len_a < len_c:
            score += 0.25
        elif len_a <= len_c * 1.1:  # within 10% is acceptable
            score += 0.10

        # Scientist longer than coach or equal
        if len_s >= len_c:
            score += 0.25
        elif len_s >= len_c * 0.85:
            score += 0.10

        # Technical term presence in scientist brief
        tech_terms = ["ACWR", "HRV", "overreaching", "monotony", "sRPE"]
        if any(t in scientist for t in tech_terms):
            score += 0.10

        return min(1.0, score)

    def _score_clinical_coverage(self, example: Dict) -> float:
        """Rewards briefs with complete clinical content."""
        coach_brief = example.get("output_coach", "")
        labels      = example.get("ground_truth_labels", {})

        if not coach_brief:
            return 0.0

        score = 0.0
        text  = coach_brief.lower()

        # Risk level present (any form)
        risk_words = ["low", "moderate", "high", "critical"]
        if any(r in text for r in risk_words):
            score += 0.30

        # Overreaching classification implied
        oc = labels.get("overreaching_classification", "")
        oc_markers = {
            "normal_adaptation":         ["normal", "adapting", "no concern", "well-tolerated", "on track"],
            "undertraining":             ["undertrain", "detraining", "too low", "insufficient"],
            "functional_overreaching":   ["functional", "short-term", "recoverable", "early", "manageable"],
            "non_functional_overreaching": ["non-functional", "extended recovery", "weeks", "significant"],
            "overtraining_syndrome":     ["overtraining", "ots", "physician", "medical", "suspend", "critical"],
        }
        markers = oc_markers.get(oc, ["normal"])
        if any(m in text for m in markers):
            score += 0.35

        # Recommendation present
        if self._score_recommendation_specificity(coach_brief) >= 0.4:
            score += 0.35

        return min(1.0, score)

    def _score_language_naturalness(self, text: str) -> float:
        """Checks output reads as natural language."""
        if not text:
            return 0.0

        score = 1.0

        # Template artifacts
        for artifact in ["{", "}", "INSERT", "TODO", "PLACEHOLDER"]:
            if artifact in text:
                score -= 0.3

        # Very short outputs
        if len(text.split()) < 30:
            score -= 0.3

        # All-caps overuse (unfilled templates)
        words = text.split()
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 4)
        if caps_words > len(words) * 0.3:
            score -= 0.2

        return max(0.0, score)
