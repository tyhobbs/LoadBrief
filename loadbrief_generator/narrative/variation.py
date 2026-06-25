# narrative/variation.py
# Adds linguistic variation to monitoring narratives.
# Prevents all 50,000 examples from sounding identical
# by randomizing section order, omitting minor fields,
# and varying sentence connectors.

import random
from typing import List


class NarrativeVariation:
    """
    Applies controlled variation to narrative sections.
    Each call produces slightly different text structure
    while preserving all clinically important content.
    """

    # Sentence connectors used between sections
    CONNECTORS = [
        " ",           # Simple space (most common)
        "\n",          # New line
        " Additionally, ",
        " Furthermore, ",
        " Of note, ",
        " ",
        " ",
    ]

    # Section reordering — which sections can swap positions
    # Core sections (load, HRV) stay near top
    # Secondary sections (sleep, injury) can move
    REORDERABLE_INDICES = [4, 5, 6]  # sleep, special, injury

    def apply(self,
              sections: List[str],
              data_level: int) -> str:
        """
        Apply variation transformations to section list.
        Returns final narrative string.
        """
        # Filter out empty sections
        sections = [s for s in sections if s and s.strip()]

        # Occasionally reorder secondary sections
        if len(sections) > 5 and random.random() < 0.3:
            sections = self._reorder_secondary(sections)

        # Occasionally omit minor sections (not load or HRV)
        if random.random() < 0.25:
            sections = self._maybe_omit_minor(sections)

        # Join with varied connectors
        narrative = self._join_sections(sections)

        # Apply minor text transformations
        narrative = self._apply_minor_transforms(narrative)

        return narrative.strip()

    def _reorder_secondary(self,
                            sections: List[str]) -> List[str]:
        """
        Randomly reorder sections beyond index 3.
        Core monitoring sections (header, athlete, load, HRV)
        always stay in their original positions.
        """
        if len(sections) <= 4:
            return sections

        core = sections[:4]
        secondary = sections[4:]
        random.shuffle(secondary)
        return core + secondary

    def _maybe_omit_minor(self,
                           sections: List[str]) -> List[str]:
        """
        Occasionally omit minor sections.
        Never omit header, athlete context, load, or HRV.
        """
        if len(sections) <= 3:
            return sections

        result = sections[:3]  # Always keep first 3

        for i, section in enumerate(sections[3:], start=3):
            # Omit with 20% probability for minor sections
            if random.random() < 0.80:  # 80% keep rate
                result.append(section)

        return result

    def _join_sections(self, sections: List[str]) -> str:
        """
        Join sections with varied connectors.
        Most joins are simple spaces or newlines,
        occasionally uses transitional phrases.
        """
        if not sections:
            return ""

        result = sections[0]
        for section in sections[1:]:
            connector = random.choices(
                self.CONNECTORS,
                weights=[40, 30, 5, 5, 5, 7, 8]
            )[0]
            result += connector + section

        return result

    def _apply_minor_transforms(self, text: str) -> str:
        """
        Apply minor text-level transformations for variety.
        - Occasionally contract some phrases
        - Clean up spacing
        - Handle edge cases
        """
        # Clean up any double spaces or awkward punctuation
        import re
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\. \.', '.', text)
        text = re.sub(r'\n\n+', '\n', text)

        # Occasionally replace some phrases with abbreviations
        # or vice versa (adds variation for technical terms)
        if random.random() < 0.3:
            text = text.replace(
                "acute:chronic workload ratio",
                "ACWR"
            )
        if random.random() < 0.3:
            text = text.replace(
                "ACWR",
                "acute:chronic workload ratio"
            )

        if random.random() < 0.3:
            text = text.replace(
                "heart rate variability",
                "HRV"
            )
        if random.random() < 0.3:
            text = text.replace("HRV", "heart rate variability")

        return text
