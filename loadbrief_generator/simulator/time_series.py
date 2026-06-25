# simulator/time_series.py
# Generates realistic day-by-day monitoring time series
# for each athlete and scenario combination

import numpy as np
import random
from typing import Dict, List, Optional


def _safe_randint(a, b):
    """randint that handles equal or inverted ranges safely"""
    a, b = int(a), int(b)
    if a >= b:
        return a
    return random.randint(a, b)


class TimeSeriesSimulator:
    """
    Simulates 28 days of athlete monitoring data.
    Each day contains: session_load, session_rpe, session_duration,
    hrv, wellness scores, is_rest_day.

    All simulation parameters are grounded in published
    exercise science literature.
    """

    def __init__(self, acwr_thresholds: dict,
                 hrv_thresholds: dict,
                 wellness_norms: dict):
        self.acwr_thresholds = acwr_thresholds
        self.hrv_thresholds = hrv_thresholds
        self.wellness_norms = wellness_norms

    def simulate(self,
                 athlete: Dict,
                 scenario,
                 weeks: int = 4) -> List[Dict]:
        """
        Main simulation entry point.
        Returns a list of daily monitoring records.
        """
        n_days = weeks * 7
        chronic_load = athlete["baseline_chronic_load"]
        baseline_hrv = athlete["baseline_hrv"]
        baseline_wellness = athlete["baseline_wellness"]

        # Generate session loads following scenario pattern
        session_loads = self._generate_load_pattern(
            scenario=scenario,
            n_days=n_days,
            chronic_load=chronic_load,
            athlete=athlete
        )

        # Generate daily records
        time_series = []
        cumulative_fatigue = 0.0

        for day_idx in range(n_days):
            load = session_loads[day_idx]
            is_rest_day = (load == 0)

            # Calculate cumulative fatigue
            # Fatigue decays with half-life of ~3 days (Banister 1991)
            cumulative_fatigue = (
                cumulative_fatigue * 0.79 + load * 0.21
            )

            # Generate each signal
            record = {
                "day": day_idx,
                "week": day_idx // 7,
                "is_rest_day": is_rest_day
            }

            # Session load signals
            if not is_rest_day:
                rpe, duration = self._session_from_load(
                    load, athlete["rpe_sensitivity"]
                )
                record["session_rpe"] = round(rpe, 1)
                record["session_duration_min"] = round(duration, 0)
                record["session_load"] = round(load, 1)
            else:
                record["session_rpe"] = None
                record["session_duration_min"] = None
                record["session_load"] = 0.0

            # HRV — responds to accumulated fatigue
            record["hrv"] = self._simulate_hrv(
                baseline_hrv=baseline_hrv,
                cumulative_fatigue=cumulative_fatigue,
                day_idx=day_idx,
                scenario=scenario,
                athlete=athlete
            )

            # Wellness — responds to fatigue with more noise
            record["wellness"] = self._simulate_wellness(
                baseline_wellness=baseline_wellness,
                cumulative_fatigue=cumulative_fatigue,
                day_idx=day_idx,
                scenario=scenario,
                athlete=athlete
            )

            # Sleep duration (hours)
            record["sleep_hours"] = self._simulate_sleep_duration(
                wellness_sleep=record["wellness"]["sleep_quality"],
                scenario=scenario,
                day_idx=day_idx
            )

            time_series.append(record)

        return time_series

    # ── Load Pattern Generators ──────────────────────────────────────

    def _generate_load_pattern(self,
                               scenario,
                               n_days: int,
                               chronic_load: float,
                               athlete: Dict) -> List[float]:
        """
        Generate daily session loads following scenario trajectory.
        Uses chronic_load as the baseline reference.
        """
        pattern_name = scenario.load_pattern
        special = scenario.special_parameters

        # Determine training days per week from athlete phase
        training_days = self._get_training_days_per_week(
            athlete["phase"], athlete["level"]
        )

        # Base weekly load = chronic_load (by definition)
        base_daily = chronic_load / training_days

        loads = []

        for day_idx in range(n_days):
            week = day_idx // 7
            day_of_week = day_idx % 7
            is_training_day = (day_of_week < training_days)

            if not is_training_day:
                loads.append(0.0)
                continue

            load = self._calculate_day_load(
                pattern_name=pattern_name,
                base_daily=base_daily,
                week=week,
                day_of_week=day_of_week,
                n_weeks=n_days // 7,
                special=special,
                training_days=training_days
            )

            # Add individual variability (±15%)
            variability = np.random.normal(1.0, 0.08)
            load = max(50, load * variability)
            loads.append(round(load, 1))

        return loads

    def _calculate_day_load(self,
                            pattern_name: str,
                            base_daily: float,
                            week: int,
                            day_of_week: int,
                            n_weeks: int,
                            special: dict,
                            training_days: int) -> float:
        """Route to specific load pattern calculator"""

        patterns = {
            "5_percent_increase_per_week":
                self._pattern_progressive,
            "spike_week3_then_maintained":
                self._pattern_spike_week3,
            "20_percent_decrease_per_week":
                self._pattern_taper,
            "consistently_low":
                self._pattern_consistently_low,
            "flat_low":
                self._pattern_flat_low,
            "constant_daily_identical":
                self._pattern_monotony,
            "normal_consistent":
                self._pattern_normal_consistent,
            "high_but_athlete_tolerating":
                self._pattern_high_tolerating,
            "graduated_return_from_zero":
                self._pattern_illness_return,
            "rapid_increase_from_low_base":
                self._pattern_preseason,
            "minimal_post_match":
                self._pattern_post_competition,
            "alternating_match_minimal_training":
                self._pattern_fixture_congestion,
            "reduced_intensity_maintained_volume":
                self._pattern_altitude,
            "match_load_accumulation":
                self._pattern_match_accumulation,
            "sustained_elevated_multiweek":
                self._pattern_early_overreaching,
            "chronically_elevated_then_declining_performance":
                self._pattern_ots,
            "normal_stable":
                self._pattern_normal_consistent,
            "elevated_sustained":
                self._pattern_high_tolerating,
            "post_match_dip":
                self._pattern_post_competition,
            "aggressive_ramp_from_zero":
                self._pattern_preseason,
            "progressively_declining_performance":
                self._pattern_early_overreaching,
            "variable_inconsistent_tracking":
                self._pattern_recreational,
            "double_session_days":
                self._pattern_double_sessions,
            "reduced_travel_period":
                self._pattern_travel,
            "normal_but_phv_risk_elevated":
                self._pattern_normal_consistent,
            "undefined_then_spiking":
                self._pattern_illness_return
        }

        fn = patterns.get(
            pattern_name, self._pattern_normal_consistent
        )
        return fn(base_daily, week, day_of_week,
                  n_weeks, special, training_days)

    def _pattern_progressive(self, base, week,
                              dow, n_weeks, special, td):
        """5% progressive overload per week"""
        multiplier = 1.0 + (week * 0.05)
        return base * multiplier

    def _pattern_spike_week3(self, base, week,
                              dow, n_weeks, special, td):
        """Normal → spike in week 3 → maintained high"""
        if week == 0:
            return base * 1.0
        elif week == 1:
            return base * 1.05
        elif week == 2:
            spike = special.get("spike_magnitude", (1.8, 2.2))
            mag = random.uniform(*spike) if isinstance(
                spike, tuple) else spike
            return base * mag
        else:
            return base * 1.5  # maintained high

    def _pattern_taper(self, base, week,
                        dow, n_weeks, special, td):
        """Progressive taper — 20% reduction per week"""
        multiplier = max(0.4, 1.0 - (week * 0.20))
        return base * multiplier

    def _pattern_consistently_low(self, base, week,
                                   dow, n_weeks, special, td):
        """Detraining — load consistently low"""
        return base * random.uniform(0.35, 0.55)

    def _pattern_flat_low(self, base, week,
                           dow, n_weeks, special, td):
        """Flat low — undertraining"""
        return base * random.uniform(0.4, 0.65)

    def _pattern_monotony(self, base, week,
                           dow, n_weeks, special, td):
        """
        Same load every day — high monotony.
        Minimal daily variation deliberately.
        """
        return base * random.uniform(0.95, 1.05)  # tiny variation

    def _pattern_normal_consistent(self, base, week,
                                    dow, n_weeks, special, td):
        """Normal consistent training — slight weekly variation"""
        weekly_var = random.uniform(0.90, 1.10)
        daily_var = random.uniform(0.85, 1.15)
        return base * weekly_var * daily_var

    def _pattern_high_tolerating(self, base, week,
                                  dow, n_weeks, special, td):
        """Elevated but tolerated — athlete handling high load"""
        return base * random.uniform(1.4, 1.7)

    def _pattern_illness_return(self, base, week,
                                 dow, n_weeks, special, td):
        """Graduated return — starts very low, builds conservatively"""
        illness_days = special.get("illness_days", (7, 14))
        if isinstance(illness_days, tuple):
            illness_days = _safe_randint(*illness_days)

        illness_weeks = illness_days / 7
        if week < illness_weeks / 7:
            return base * 0.1  # near zero during illness
        else:
            recovery_week = week - (illness_weeks / 7)
            ramp = min(1.0, recovery_week * 0.25)
            return base * ramp

    def _pattern_preseason(self, base, week,
                            dow, n_weeks, special, td):
        """
        Preseason ramp — starts at off-season base (low),
        rapidly increases.
        """
        off_season_factor = 0.5
        ramp_per_week = 0.15
        current = off_season_factor + (week * ramp_per_week)
        return base * min(1.3, current)

    def _pattern_post_competition(self, base, week,
                                   dow, n_weeks, special, td):
        """Post-match — very low load for several days"""
        days_post = special.get("days_post_match", (1, 4))
        if isinstance(days_post, tuple):
            days_post = _safe_randint(*days_post)

        total_day = week * 7 + dow
        if total_day < days_post:
            return base * random.uniform(0.1, 0.25)
        else:
            return base * random.uniform(0.7, 1.0)

    def _pattern_fixture_congestion(self, base, week,
                                     dow, n_weeks, special, td):
        """
        Match congestion — alternating match loads
        and minimal training between.
        """
        matches_per_week = special.get("matches_per_week", 2)
        if isinstance(matches_per_week, tuple): matches_per_week = _safe_randint(*matches_per_week)
        # Match days get high load, other days very low
        if dow in range(matches_per_week):
            return base * random.uniform(1.5, 2.0)  # match load
        else:
            return base * random.uniform(0.15, 0.35)  # recovery

    def _pattern_altitude(self, base, week,
                           dow, n_weeks, special, td):
        """
        Altitude camp — reduced intensity but maintained volume.
        Physiological cost higher than load suggests.
        """
        return base * random.uniform(0.75, 0.90)

    def _pattern_match_accumulation(self, base, week,
                                     dow, n_weeks, special, td):
        """Accumulating match load"""
        match_days = special.get("matches_per_week", 2)
        if isinstance(match_days, tuple): match_days = _safe_randint(*match_days)
        if dow < match_days:
            return base * random.uniform(1.3, 1.7)
        return base * random.uniform(0.2, 0.4)

    def _pattern_early_overreaching(self, base, week,
                                     dow, n_weeks, special, td):
        """
        Sustained elevated load driving overreaching.
        Load stays high for multiple weeks.
        """
        return base * random.uniform(1.3, 1.6)

    def _pattern_ots(self, base, week,
                      dow, n_weeks, special, td):
        """
        OTS pattern — historically very high, now forced reduction
        but damage already done.
        """
        if week <= 2:
            return base * random.uniform(1.6, 2.0)
        else:
            # Forced reduction
            return base * random.uniform(0.7, 0.9)

    def _pattern_recreational(self, base, week,
                               dow, n_weeks, special, td):
        """Irregular recreational pattern with gaps"""
        if random.random() < 0.15:  # 15% chance of missed session
            return 0.0
        return base * random.uniform(0.6, 1.4)

    def _pattern_double_sessions(self, base, week,
                                  dow, n_weeks, special, td):
        """
        Double sessions on some days.
        Total daily load higher but split across two sessions.
        """
        double_days = special.get("double_days_per_week", 3)
        if isinstance(double_days, tuple): double_days = _safe_randint(*double_days)
        if dow < double_days:
            return base * random.uniform(1.6, 2.0)
        return base * random.uniform(0.8, 1.1)

    def _pattern_travel(self, base, week,
                         dow, n_weeks, special, td):
        """Minimal load during travel period"""
        return base * random.uniform(0.3, 0.6)

    # ── Signal Generators ────────────────────────────────────────────

    def _simulate_hrv(self,
                      baseline_hrv: float,
                      cumulative_fatigue: float,
                      day_idx: int,
                      scenario,
                      athlete: Dict) -> Optional[float]:
        """
        Simulate morning HRV.
        HRV responds to cumulative fatigue and scenario-specific
        patterns. Individual reactivity modulates the response.
        """
        hrv_pattern = scenario.hrv_pattern
        special = scenario.special_parameters
        reactivity = athlete["hrv_reactivity"]

        # Base HRV suppression from cumulative fatigue
        # Normalized to chronic load to get relative fatigue
        chronic = athlete["baseline_chronic_load"]
        fatigue_ratio = cumulative_fatigue / max(chronic, 1)
        base_suppression = (fatigue_ratio - 1.0) * 10 * reactivity

        # Pattern-specific modifiers
        pattern_modifier = self._hrv_pattern_modifier(
            hrv_pattern, day_idx, scenario, special
        )

        # Daily noise
        noise = np.random.normal(0, 3.5)

        hrv = baseline_hrv + base_suppression + pattern_modifier + noise
        return round(max(15, min(180, hrv)), 1)

    def _hrv_pattern_modifier(self, pattern: str,
                               day_idx: int,
                               scenario,
                               special: dict) -> float:
        """Additional HRV modifier based on scenario pattern"""

        if pattern == "stable_with_positive_trend":
            return day_idx * 0.1  # slight upward trend

        elif pattern == "suppressed_following_spike":
            spike_day = 14  # spike in week 3
            if day_idx < spike_day:
                return 0.0
            else:
                days_after = day_idx - spike_day
                return -min(15, days_after * 1.5)

        elif pattern == "elevating_during_taper":
            return day_idx * 0.3  # clear upward trend in taper

        elif pattern == "altitude_suppression_then_adapting":
            if day_idx < 5:
                return -12  # initial suppression
            elif day_idx < 14:
                return -12 + (day_idx - 5) * 0.8  # gradual recovery
            else:
                return 0.0  # adapted

        elif pattern == "jet_lag_circadian_disruption":
            zones = special.get("time_zones_crossed", 8)
            if isinstance(zones, tuple): zones = _safe_randint(*zones)
            disruption_days = min(zones * 0.5, 5)
            if day_idx < disruption_days:
                return -zones * 0.8
            else:
                recovery = (day_idx - disruption_days) * 1.5
                return max(-zones * 0.8, -zones * 0.8 + recovery)

        elif pattern == "heat_cardiovascular_suppression":
            temp = special.get("temperature_celsius", 38)
            if isinstance(temp, tuple): temp = _safe_randint(*temp)
            heat_effect = -(temp - 25) * 0.4
            return max(-12, heat_effect)

        elif pattern == "progressively_suppressing_multiweek":
            return -day_idx * 0.4  # steady downward trend

        elif pattern == "chronically_suppressed":
            return -15 + np.random.normal(0, 2)

        elif pattern == "elevated_good_recovery":
            return 8 + np.random.normal(0, 2)

        elif pattern == "suppressed_non_load_cause":
            # Non-load stressor — constant suppression
            return -10 + np.random.normal(0, 2)

        elif pattern == "stable_despite_high_acwr":
            return np.random.normal(0, 2)  # normal despite high load

        elif pattern == "recovering_but_variable":
            if day_idx < 7:
                return -8 + np.random.normal(0, 4)
            else:
                return -8 + day_idx * 0.3

        elif pattern == "not_available":
            return float('nan')  # data not available

        elif pattern == "post_match_suppression":
            days_post = special.get("days_post_match", 2)
            if isinstance(days_post, tuple):
                days_post = _safe_randint(*days_post)
            if day_idx < days_post:
                return -10
            return 0.0

        else:
            return np.random.normal(0, 2)

    def _simulate_wellness(self,
                           baseline_wellness: Dict,
                           cumulative_fatigue: float,
                           day_idx: int,
                           scenario,
                           athlete: Dict) -> Dict:
        """
        Simulate daily wellness questionnaire responses.
        Each dimension responds differently to fatigue.
        """
        wellness_pattern = scenario.wellness_pattern
        special = scenario.special_parameters
        chronic = athlete["baseline_chronic_load"]
        fatigue_ratio = cumulative_fatigue / max(chronic, 1)

        # Pattern modifier affects all wellness dimensions
        pattern_mod = self._wellness_pattern_modifier(
            wellness_pattern, day_idx, scenario, special
        )

        wellness = {}
        for dim, baseline in baseline_wellness.items():
            # Fatigue effect is different per dimension
            fatigue_effects = {
                "sleep_quality":   -fatigue_ratio * 0.8,
                "fatigue":          fatigue_ratio * 1.2,
                "muscle_soreness":  fatigue_ratio * 1.0,
                "mood":            -fatigue_ratio * 0.6,
                "stress":           fatigue_ratio * 0.5
            }
            fatigue_effect = fatigue_effects.get(dim, 0)

            # Dimension-specific pattern modifier
            dim_mod = pattern_mod.get(dim, pattern_mod.get("all", 0))

            # Random daily noise
            noise = np.random.normal(0, 0.25)

            raw = baseline + fatigue_effect + dim_mod + noise

            # Clip to valid scale range
            wellness[dim] = round(np.clip(raw, 1.0, 5.0), 1)

        return wellness

    def _wellness_pattern_modifier(self, pattern: str,
                                   day_idx: int,
                                   scenario,
                                   special: dict) -> Dict:
        """
        Returns per-dimension wellness modifiers for each pattern.
        Positive values improve wellness for positive dims (sleep, mood),
        negative values worsen them.
        For inverse dims (fatigue, soreness, stress): positive = worse.
        """

        if pattern == "stable_good":
            return {"all": 0.0}

        elif pattern == "declining_following_spike":
            spike_day = 14
            if day_idx < spike_day:
                return {"all": 0.0}
            days_after = min(day_idx - spike_day, 10)
            severity = days_after * 0.12
            return {
                "sleep_quality": -severity,
                "fatigue": severity,
                "muscle_soreness": severity * 1.2,
                "mood": -severity * 0.8,
                "stress": severity * 0.6
            }

        elif pattern == "improving_during_taper":
            improvement = day_idx * 0.05
            return {
                "sleep_quality": improvement,
                "fatigue": -improvement,
                "muscle_soreness": -improvement,
                "mood": improvement,
                "stress": -improvement
            }

        elif pattern == "stable_low_fatigue":
            return {
                "sleep_quality": 0.3,
                "fatigue": -0.5,
                "muscle_soreness": -0.5,
                "mood": 0.2,
                "stress": -0.3
            }

        elif pattern == "severe_crash_non_load":
            return {
                "sleep_quality": -1.5,
                "fatigue": 1.5,
                "muscle_soreness": 0.5,
                "mood": -1.5,
                "stress": 1.5
            }

        elif pattern == "normal_despite_high_acwr":
            return {"all": 0.0}  # athlete handling it fine

        elif pattern == "progressively_declining_multiweek":
            severity = day_idx * 0.06
            return {
                "sleep_quality": -severity,
                "fatigue": severity,
                "muscle_soreness": severity,
                "mood": -severity * 0.8,
                "stress": severity * 0.7
            }

        elif pattern == "severely_depressed_all_dimensions":
            return {
                "sleep_quality": -1.8,
                "fatigue": 2.0,
                "muscle_soreness": 1.5,
                "mood": -1.8,
                "stress": 1.8
            }

        elif pattern == "post_match_fatigue":
            days_post = special.get("days_post_match", 2)
            if isinstance(days_post, tuple):
                days_post = _safe_randint(*days_post)
            if day_idx < days_post:
                return {
                    "sleep_quality": -0.5,
                    "fatigue": 1.2,
                    "muscle_soreness": 1.5,
                    "mood": -0.3,
                    "stress": 0.3
                }
            return {"all": 0.0}

        elif pattern == "altitude_stress_then_improving":
            if day_idx < 5:
                return {
                    "sleep_quality": -1.0,
                    "fatigue": 1.2,
                    "muscle_soreness": 0.8,
                    "mood": -0.6,
                    "stress": 0.8
                }
            improvement = (day_idx - 5) * 0.08
            return {
                "sleep_quality": -1.0 + improvement,
                "fatigue": 1.2 - improvement,
                "muscle_soreness": 0.8 - improvement,
                "mood": -0.6 + improvement * 0.5,
                "stress": 0.8 - improvement * 0.5
            }

        elif pattern == "heat_stress_response":
            temp = special.get("temperature_celsius", 38)
            if isinstance(temp, tuple): temp = _safe_randint(*temp)
            heat_effect = (temp - 25) * 0.06
            return {
                "sleep_quality": -heat_effect * 0.8,
                "fatigue": heat_effect,
                "muscle_soreness": heat_effect * 0.6,
                "mood": -heat_effect * 0.7,
                "stress": heat_effect * 0.5
            }

        elif pattern == "sleep_disrupted_travel":
            zones = special.get("time_zones_crossed", 8)
            if isinstance(zones, tuple): zones = _safe_randint(*zones)
            disruption = min(zones * 0.12, 1.2)
            return {
                "sleep_quality": -disruption,
                "fatigue": disruption,
                "muscle_soreness": 0.2,
                "mood": -disruption * 0.6,
                "stress": disruption * 0.5
            }

        elif pattern == "fatigue_accumulating_between_sessions":
            severity = day_idx * 0.04
            return {
                "sleep_quality": -severity * 0.8,
                "fatigue": severity * 1.2,
                "muscle_soreness": severity,
                "mood": -severity * 0.6,
                "stress": severity * 0.4
            }

        elif pattern == "moderate_fatigue_accumulation":
            severity = day_idx * 0.03
            return {
                "sleep_quality": -severity,
                "fatigue": severity,
                "muscle_soreness": severity * 0.8,
                "mood": -severity * 0.5,
                "stress": severity * 0.3
            }

        elif pattern == "improving_but_fragile":
            if day_idx < 7:
                return {
                    "sleep_quality": -0.8,
                    "fatigue": 1.0,
                    "muscle_soreness": 0.5,
                    "mood": -0.6,
                    "stress": 0.4
                }
            recovery = (day_idx - 7) * 0.06
            return {
                "sleep_quality": -0.8 + recovery,
                "fatigue": 1.0 - recovery,
                "muscle_soreness": 0.5 - recovery * 0.5,
                "mood": -0.6 + recovery * 0.7,
                "stress": 0.4 - recovery * 0.3
            }

        elif pattern == "growing_pains_present":
            return {
                "sleep_quality": -0.3,
                "fatigue": 0.4,
                "muscle_soreness": 1.0,  # Growing pains
                "mood": -0.2,
                "stress": 0.2
            }

        elif pattern == "subjective_only":
            return {"all": np.random.normal(0, 0.3)}

        elif pattern == "variable_growth_related":
            noise_scale = 0.5
            return {
                dim: np.random.normal(0, noise_scale)
                for dim in ["sleep_quality", "fatigue",
                            "muscle_soreness", "mood", "stress"]
            }

        else:
            return {"all": 0.0}

    def _session_from_load(self, load: float,
                           rpe_sensitivity: float):
        """
        Back-calculate session RPE and duration from load.
        Load = RPE × Duration (Foster 1998)
        """
        # Duration varies between 45-120 minutes
        duration = random.uniform(45, 120)
        rpe = load / duration

        # Apply individual sensitivity and clip to 1-10 scale
        rpe = rpe * rpe_sensitivity
        rpe = np.clip(rpe, 1.0, 10.0)

        # Recalculate duration to maintain load
        duration = load / rpe

        return rpe, duration

    def _simulate_sleep_duration(self,
                                  wellness_sleep: float,
                                  scenario,
                                  day_idx: int) -> float:
        """
        Simulate sleep duration in hours.
        Correlates with sleep quality wellness score.
        """
        # Base sleep ~7.5 hours, ±1 based on wellness
        base_sleep = 7.5
        wellness_effect = (wellness_sleep - 3.0) * 0.4
        noise = np.random.normal(0, 0.5)

        sleep = base_sleep + wellness_effect + noise

        # Travel pattern reduces sleep
        if "travel" in scenario.hrv_pattern.lower():
            zones = scenario.special_parameters.get(
                "time_zones_crossed", 8
            )
            if day_idx < zones * 0.3:
                sleep -= random.uniform(1, 2.5)

        return round(np.clip(sleep, 3.0, 12.0), 1)

    def _get_training_days_per_week(self, phase: str,
                                     level: str) -> int:
        """Determine training days per week from phase and level"""
        phase_days = {
            "off_season": 3,
            "pre_season_early": 4,
            "pre_season_late": 5,
            "in_season_early": 4,
            "in_season_mid": 4,
            "in_season_late": 4,
            "taper": 3,
            "competition": 3,
            "post_season_recovery": 2,
            "return_from_injury": 3,
            "return_from_illness": 2
        }
        base_days = phase_days.get(phase, 4)

        # Elite athletes train more days
        level_bonus = {
            "recreational": -1,
            "amateur": 0,
            "semi_professional": 0,
            "professional": 1,
            "elite": 1
        }.get(level, 0)

        return max(2, min(6, base_days + level_bonus))
