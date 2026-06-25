# narrative/narrator.py
# Converts monitoring metrics and time series into
# the natural language input narrative for each training example.
# This generates the INPUT side of every training pair.

import random
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional

from narrative.templates import (
    HEADER_TEMPLATES, ATHLETE_CONTEXT_TEMPLATES,
    ACWR_TEMPLATES, LOAD_VALUE_TEMPLATES, MONOTONY_TEMPLATES,
    HRV_TEMPLATES, WELLNESS_ALL_NORMAL, WELLNESS_ONE_FLAG,
    WELLNESS_MULTIPLE_FLAGS, WELLNESS_PARTIAL_DATA,
    RPE_LOG_TEMPLATES, SLEEP_TEMPLATES,
    INJURY_NOTE_TEMPLATES, ALTITUDE_NOTE_TEMPLATES,
    HEAT_NOTE_TEMPLATES, TRAVEL_NOTE_TEMPLATES,
    YOUTH_NOTE_TEMPLATES
)
from narrative.variation import NarrativeVariation


class MonitoringNarrator:
    """
    Converts structured monitoring data into natural language narratives.
    Produces varied, realistic text that reads like actual
    sports science monitoring reports.
    """

    def __init__(self):
        self.variation = NarrativeVariation()

    def generate(self,
                 athlete: Dict,
                 time_series: List[Dict],
                 metrics: Dict,
                 data_level: int,
                 scenario) -> str:
        """
        Main entry point — generates a complete monitoring narrative.
        """
        sections = []

        # 1. Header with time period
        sections.append(self._build_header(athlete, time_series))

        # 2. Athlete context
        sections.append(
            self._build_athlete_context(athlete)
        )

        # 3. Training load section (always available)
        sections.append(
            self._build_load_section(
                metrics["acwr"], athlete, data_level
            )
        )

        # 4. HRV section (Level 2+)
        if metrics["hrv"].get("available", False):
            sections.append(
                self._build_hrv_section(metrics["hrv"])
            )
        elif data_level >= 2:
            sections.append(
                self._build_hrv_unavailable_section()
            )

        # 5. Wellness section
        sections.append(
            self._build_wellness_section(
                metrics["wellness"], data_level
            )
        )

        # 6. Sleep section
        sections.append(
            self._build_sleep_section(time_series, data_level)
        )

        # 7. Special circumstance notes
        special_note = self._build_special_notes(
            scenario, athlete
        )
        if special_note:
            sections.append(special_note)

        # 8. Injury history note (sometimes)
        if (athlete["injury_history"]["has_prior_injury"] and
                random.random() < 0.6):
            sections.append(
                self._build_injury_note(athlete)
            )

        # Apply variation — randomly reorder non-essential sections,
        # occasionally omit minor fields
        narrative = self.variation.apply(sections, data_level)

        return narrative

    def _build_header(self, athlete: Dict,
                      time_series: List[Dict]) -> str:
        """Build monitoring period header"""
        # Generate a plausible monitoring period
        end_date = date.today() - timedelta(
            days=random.randint(0, 30)
        )
        start_date = end_date - timedelta(days=len(time_series))

        period_formats = [
            f"{start_date.strftime('%B %d')} – "
            f"{end_date.strftime('%B %d, %Y')}",
            f"past {len(time_series)} days",
            f"{len(time_series)//7}-week monitoring block",
            f"weeks {random.randint(1,20)}–"
            f"{random.randint(21,35)} of season"
        ]

        period = random.choice(period_formats)
        template = random.choice(HEADER_TEMPLATES)
        sport_display = athlete["sport"].replace("_", " ").title()

        return template.format(
            period=period,
            sport=sport_display
        )

    def _build_athlete_context(self, athlete: Dict) -> str:
        """Build athlete profile context sentence"""
        template = random.choice(ATHLETE_CONTEXT_TEMPLATES)
        sport_display = athlete["sport"].replace("_", " ").title()
        position_display = athlete["position"].replace(
            "_", " "
        ).title()
        phase_display = athlete["phase"].replace("_", " ")
        level_display = athlete["level"].replace("_", " ")

        return template.format(
            sport=sport_display,
            position=position_display,
            level=level_display,
            training_age=f"{athlete['training_age_years']:.0f}",
            phase=phase_display
        )

    def _build_load_section(self, acwr_metrics: Dict,
                             athlete: Dict,
                             data_level: int) -> str:
        """Build training load section"""
        if not acwr_metrics.get("available", True):
            return ("Training load data insufficient for ACWR "
                    "calculation.")

        acwr = acwr_metrics.get("acwr")
        zone = acwr_metrics.get("zone", "sweet_spot")
        acute = acwr_metrics.get("acute_load", 0)
        chronic = acwr_metrics.get("chronic_load", 0)
        monotony = acwr_metrics.get("monotony", 0)
        strain = acwr_metrics.get("strain", 0)

        parts = []

        # ACWR description
        if acwr is not None:
            acwr_templates = ACWR_TEMPLATES.get(
                zone, ACWR_TEMPLATES["sweet_spot"]
            )
            acwr_text = random.choice(acwr_templates).format(
                acwr=acwr
            )
            parts.append(acwr_text)

            # Raw load values (for Level 2+ athletes)
            if data_level >= 2 and random.random() < 0.7:
                load_template = random.choice(LOAD_VALUE_TEMPLATES)
                parts.append(load_template.format(
                    acute=acute, chronic=chronic
                ))
        else:
            parts.append(random.choice(
                ACWR_TEMPLATES["insufficient_data"]
            ))

        # Monotony (for Level 2+ and sometimes Level 1)
        if monotony > 0 and (
            data_level >= 2 or random.random() < 0.3
        ):
            if monotony > 2.0:
                mono_zone = "high_risk" if monotony > 3.0 \
                            else "concerning"
                mono_template = random.choice(
                    MONOTONY_TEMPLATES[mono_zone]
                )
                parts.append(mono_template.format(
                    monotony=monotony
                ))
            elif random.random() < 0.4:
                # Occasionally mention normal monotony
                parts.append(random.choice(
                    MONOTONY_TEMPLATES["acceptable"]
                ).format(monotony=monotony))

        return " ".join(parts)

    def _build_hrv_section(self, hrv_analysis: Dict) -> str:
        """Build HRV section"""
        status = hrv_analysis.get("status", "normal")
        current_delta = hrv_analysis.get(
            "current_vs_baseline_ms", 0
        )
        mean_delta = hrv_analysis.get(
            "7day_mean_vs_baseline_ms", 0
        )
        baseline = hrv_analysis.get("baseline", 65)
        days_suppressed = hrv_analysis.get(
            "consecutive_suppressed_days", 0
        )
        recent_values = hrv_analysis.get("recent_values", [])
        trend = hrv_analysis.get("trend_direction", "stable")

        # Current HRV value
        current_hrv = baseline + current_delta
        delta_abs = abs(mean_delta)

        templates = HRV_TEMPLATES.get(
            status, HRV_TEMPLATES["normal"]
        )
        template = random.choice(templates)

        hrv_text = template.format(
            current=round(current_hrv, 1),
            delta=round(delta_abs, 1),
            baseline=round(baseline, 1),
            days=days_suppressed
        )

        # Add trend for Level 3+
        if recent_values and random.random() < 0.5:
            trend_phrases = {
                "improving": " HRV trend is improving.",
                "declining": " HRV trend is declining.",
                "stable": " HRV trend is stable."
            }
            hrv_text += trend_phrases.get(trend, "")

        return hrv_text

    def _build_hrv_unavailable_section(self) -> str:
        """Build HRV unavailable note"""
        return random.choice(HRV_TEMPLATES["data_unavailable"])

    def _build_wellness_section(self, wellness_analysis: Dict,
                                 data_level: int) -> str:
        """Build wellness questionnaire section"""
        if not wellness_analysis.get("available", False):
            return ""

        if wellness_analysis.get("partial", False):
            return random.choice(WELLNESS_PARTIAL_DATA)

        composite = wellness_analysis.get(
            "composite_status", "good"
        )
        dimensions = wellness_analysis.get("dimensions", {})
        red_flags = wellness_analysis.get("red_flags", 0)
        amber_flags = wellness_analysis.get("amber_flags", 0)

        problem_dims = [
            (dim, data) for dim, data in dimensions.items()
            if data.get("zone") in ["red", "amber"]
        ]

        if not problem_dims:
            return random.choice(WELLNESS_ALL_NORMAL)

        elif len(problem_dims) == 1:
            dim, data = problem_dims[0]
            template = random.choice(WELLNESS_ONE_FLAG)
            return template.format(
                dim=dim.replace("_", " "),
                zone=data.get("zone", "amber"),
                trend=data.get("trend", "stable")
            )

        else:
            dim_names = ", ".join([
                d[0].replace("_", " ")
                for d in problem_dims
            ])
            composite_display = composite.replace(
                "_", " "
            )

            template = random.choice(WELLNESS_MULTIPLE_FLAGS)
            return template.format(
                dims=dim_names,
                composite=composite_display,
                n=len(problem_dims),
                red_count=red_flags,
                amber_count=amber_flags
            )

    def _build_sleep_section(self, time_series: List[Dict],
                              data_level: int) -> str:
        """Build sleep section"""
        sleep_values = [
            d["sleep_hours"] for d in time_series[-7:]
            if d.get("sleep_hours") is not None
        ]

        if not sleep_values:
            return ""

        avg_sleep = np.mean(sleep_values)

        if avg_sleep >= 7.0:
            category = "good"
        elif avg_sleep >= 6.0:
            category = "borderline"
        else:
            category = "poor"

        template = random.choice(SLEEP_TEMPLATES[category])

        # Only include sleep section sometimes for variety
        if category == "good" and random.random() < 0.4:
            return ""  # Omit when sleep is fine

        return template.format(hours=avg_sleep)

    def _build_rpe_log(self, time_series: List[Dict]) -> str:
        """Build RPE log for past 7 days"""
        recent = time_series[-7:]
        rpe_values = []

        for day in recent:
            rpe = day.get("session_rpe")
            if rpe is not None:
                rpe_values.append(f"{rpe:.1f}")
            else:
                rpe_values.append("REST")

        rpe_str = ", ".join(rpe_values)
        template = random.choice(RPE_LOG_TEMPLATES)
        return template.format(rpe_log=rpe_str)

    def _build_special_notes(self, scenario,
                              athlete: Dict) -> Optional[str]:
        """Build special circumstance notes based on scenario"""
        special = scenario.special_parameters

        if "altitude_meters" in special:
            altitude = special["altitude_meters"]
            if isinstance(altitude, tuple):
                altitude = random.randint(*altitude)
            template = random.choice(ALTITUDE_NOTE_TEMPLATES)
            return template.format(altitude=altitude)

        elif "temperature_celsius" in special:
            temp = special["temperature_celsius"]
            humidity = special.get("humidity_percent", (60, 80))
            if isinstance(temp, tuple):
                temp = random.randint(*temp)
            if isinstance(humidity, tuple):
                humidity = random.randint(*humidity)
            template = random.choice(HEAT_NOTE_TEMPLATES)
            return template.format(
                temp=temp, humidity=humidity
            )

        elif "time_zones_crossed" in special:
            zones = special["time_zones_crossed"]
            direction = special.get(
                "direction", ["eastward", "westward"]
            )
            if isinstance(zones, tuple):
                zones = random.randint(*zones)
            if isinstance(direction, list):
                direction = random.choice(direction)
            template = random.choice(TRAVEL_NOTE_TEMPLATES)
            return template.format(
                zones=zones, direction=direction
            )

        elif special.get("phv_status") == "at_phv":
            return random.choice(YOUTH_NOTE_TEMPLATES)

        elif "illness_days" in special:
            illness_days = special["illness_days"]
            if isinstance(illness_days, tuple):
                illness_days = random.randint(*illness_days)
            illness_type = random.choice(
                special.get("illness_type",
                             ["viral illness"])
            )
            return (
                f"Context: athlete returning after "
                f"{illness_days}-day {illness_type.replace('_', ' ')} "
                f"enforced rest period."
            )

        return None

    def _build_injury_note(self, athlete: Dict) -> str:
        """Build injury history note"""
        sites = athlete["injury_history"].get("sites", [])
        if not sites:
            return ""

        sites_display = " and ".join([
            s.replace("_", " ") for s in sites[:2]
        ])
        template = random.choice(INJURY_NOTE_TEMPLATES)
        return template.format(sites=sites_display)
