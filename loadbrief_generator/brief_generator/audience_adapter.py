# brief_generator/audience_adapter.py
# Adjusts brief language, depth, and vocabulary
# for each of the four audience types.

import re
from typing import Dict


# ── Vocabulary substitution maps ──────────────────────────────────────
ATHLETE_VOCAB = {
    "ACWR":                          "training spike ratio",
    "acute:chronic workload ratio":  "training spike ratio",
    "HRV":                           "recovery score",
    "heart rate variability":        "recovery score",
    "AU":                            "training units",
    "arbitrary units":               "training units",
    "non_functional_overreaching":   "overtraining warning",
    "non-functional overreaching":   "overtraining warning",
    "functional overreaching":       "early overtraining signs",
    "overtraining syndrome":         "severe overtraining",
    "autonomic nervous system":      "recovery system",
    "physiological":                 "physical",
    "HRV suppression":               "low recovery score",
    "monotony index":                "training variety score",
    "sRPE":                          "how hard sessions felt",
    "RPE":                           "effort level",
    "chronic load":                  "fitness base",
    "acute load":                    "recent training",
    "detraining":                    "loss of fitness",
    "PHV":                           "growth phase",
    "Peak Height Velocity":          "growth phase"
}

COACH_VOCAB = {
    # Coaches get technical terms but not full academic language
    "non_functional_overreaching":   "non-functional overreaching",
    "arbitrary units":               "AU",
    "acute:chronic workload ratio":  "ACWR"
}

SPORTS_SCIENTIST_VOCAB = {
    # Scientists get full technical language — no substitutions
    # but we do expand some abbreviations for clarity
    "ACWR": "ACWR (acute:chronic workload ratio)"
    if False else "ACWR"  # no change
}

# ── Length targets ────────────────────────────────────────────────────
LENGTH_TARGETS = {
    "athlete":        (150, 250),
    "coach":          (250, 400),
    "sports_scientist": (350, 600),
    "api":            (0, 0)   # structured, not prose
}

# ── Tone markers ──────────────────────────────────────────────────────
TONE_OPENERS = {
    "athlete": [
        "Here is what your monitoring data is telling you: ",
        "Your training data this period: ",
        "What your numbers mean for you: "
    ],
    "coach": [
        "",  # Coaches get direct briefing without opener
        "Load management brief: ",
        ""
    ],
    "sports_scientist": [
        "",  # Scientists get pure technical content
        ""
    ]
}


class AudienceAdapter:
    """
    Transforms a completed brief for the target audience.
    Adjusts vocabulary, length, and tone without
    changing the underlying clinical content.
    """

    def __init__(self, audience_profiles: dict):
        self.profiles = audience_profiles

    def adapt(self, brief_text: str,
              audience: str,
              synthesis: Dict) -> str:
        """
        Main entry point.
        Applies all audience-specific transformations.
        """
        if audience == "api":
            return self._adapt_for_api(synthesis)

        # Apply vocabulary substitutions
        adapted = self._apply_vocabulary(brief_text, audience)

        # Apply length trimming or expansion
        adapted = self._apply_length_target(adapted, audience)

        # Apply tone adjustments
        adapted = self._apply_tone(adapted, audience)

        return adapted

    def _apply_vocabulary(self, text: str,
                           audience: str) -> str:
        """Replace technical terms based on audience level"""
        vocab_map = {
            "athlete": ATHLETE_VOCAB,
            "coach": COACH_VOCAB,
            "sports_scientist": {}
        }.get(audience, {})

        for technical, plain in vocab_map.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(technical),
                                 re.IGNORECASE)
            text = pattern.sub(plain, text)

        return text

    def _apply_length_target(self, text: str,
                              audience: str) -> str:
        """
        Adjust text length to target range.
        Athletes get shorter briefs, scientists get longer.
        """
        target_min, target_max = LENGTH_TARGETS.get(
            audience, (200, 400)
        )
        words = text.split()
        current_length = len(words)

        if current_length > target_max:
            # Trim to target max
            # Remove optional sections first (monitoring priorities
            # section is least critical for athletes)
            if audience == "athlete":
                text = self._trim_for_athlete(text, target_max)
            else:
                # For other audiences, just trim at word boundary
                text = " ".join(words[:target_max]) + "..."

        # Note: we do not artificially expand short texts
        # — shorter is fine if content is complete

        return text

    def _trim_for_athlete(self, text: str,
                           target_words: int) -> str:
        """
        Smart trimming for athlete audience.
        Keeps Risk Level, Status, and Recommendations.
        Removes detailed signal analysis if needed.
        """
        # Split into sections
        lines = text.split("\n\n")

        # Priority: RISK LEVEL > STATUS/CLASSIFICATION >
        #           RECOMMENDATIONS > SIGNAL ANALYSIS > REST
        priority_sections = []
        other_sections = []

        for line in lines:
            if any(marker in line.upper() for marker in [
                "RISK LEVEL", "CURRENT STATUS",
                "RECOMMENDATION", "SEEK SUPPORT"
            ]):
                priority_sections.append(line)
            else:
                other_sections.append(line)

        # Build from priority sections first
        result_sections = priority_sections.copy()
        word_count = len(" ".join(result_sections).split())

        # Add other sections if space allows
        for section in other_sections:
            section_words = len(section.split())
            if word_count + section_words <= target_words:
                result_sections.append(section)
                word_count += section_words

        return "\n\n".join(result_sections)

    def _apply_tone(self, text: str, audience: str) -> str:
        """Apply audience-appropriate tone adjustments"""

        if audience == "athlete":
            # Make language more personal
            text = text.replace(
                "The athlete", "You"
            ).replace(
                "the athlete", "you"
            ).replace(
                "The subject", "You"
            )

            # Remove overly clinical phrasing
            text = text.replace(
                "Classification based on Meeusen et al. (2013) criteria.",
                ""
            )

        elif audience == "sports_scientist":
            # Add literature references where appropriate
            text = text.replace(
                "ACWR in the danger zone",
                "ACWR in the danger zone (Gabbett 2016)"
            ).replace(
                "non-functional overreaching",
                "non-functional overreaching (Meeusen et al. 2013)"
            )

        # Clean up any double spaces from substitutions
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)

        return text.strip()

    def _adapt_for_api(self, synthesis: Dict) -> Dict:
        """
        Return structured JSON-compatible dict for API audience.
        Used by applications consuming the model output.
        """
        return {
            "risk_level": synthesis.get("overall_risk", "low"),
            "overreaching_classification": synthesis.get(
                "overreaching_class", "normal_adaptation"
            ),
            "confidence": synthesis.get("confidence", "high"),
            "signal_agreement": synthesis.get(
                "signal_agreement", "full"
            ),
            "primary_driver": synthesis.get(
                "primary_driver",
                {"driver": "none"}
            ),
            "conflicts": synthesis.get("all_conflicts", []),
            "recommendation_category": synthesis.get(
                "recommendation_category", "no_change"
            ),
            "synthesis_narrative": synthesis.get(
                "synthesis_narrative", ""
            ),
            "complexity_tier": synthesis.get(
                "complexity_tier", 1
            )
        }
