# simulator/scenarios.py - Scenario definitions
# (truncated stub — full version provided in previous message)
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class ScenarioConfig:
    name: str
    description: str
    complexity_tier: int
    acwr_trajectory: str
    acwr_target_final: Tuple
    hrv_pattern: str
    wellness_pattern: str
    load_pattern: str
    conflicting_signals: bool
    signal_conflicts: List[str]
    risk_level: str
    overreaching_class: str
    special_parameters: dict = field(default_factory=dict)

SCENARIO_NORMAL_PROGRESSIVE = ScenarioConfig(
    name="normal_progressive", description="Normal progressive load.",
    complexity_tier=1, acwr_trajectory="gradual_increase",
    acwr_target_final=(0.9, 1.2), hrv_pattern="stable_with_positive_trend",
    wellness_pattern="stable_good", load_pattern="5_percent_increase_per_week",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="low", overreaching_class="normal_adaptation"
)
SCENARIO_ACWR_SPIKE = ScenarioConfig(
    name="acwr_spike", description="Single week load spike.",
    complexity_tier=1, acwr_trajectory="sudden_spike_week3",
    acwr_target_final=(1.5, 2.0), hrv_pattern="suppressed_following_spike",
    wellness_pattern="declining_following_spike", load_pattern="spike_week3_then_maintained",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="high", overreaching_class="functional_overreaching",
    special_parameters={"spike_magnitude": (1.8, 2.5)}
)
SCENARIO_TAPER = ScenarioConfig(
    name="taper", description="Planned taper before competition.",
    complexity_tier=1, acwr_trajectory="progressive_decrease",
    acwr_target_final=(0.5, 0.75), hrv_pattern="elevating_during_taper",
    wellness_pattern="improving_during_taper", load_pattern="20_percent_decrease_per_week",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="low", overreaching_class="normal_adaptation",
    special_parameters={"competition_days_out": (3, 7)}
)
SCENARIO_UNDERTRAINING = ScenarioConfig(
    name="undertraining", description="Load too low, detraining risk.",
    complexity_tier=1, acwr_trajectory="flat_low",
    acwr_target_final=(0.3, 0.7), hrv_pattern="elevated_detraining",
    wellness_pattern="stable_low_fatigue", load_pattern="consistently_low",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="low_with_detraining_flag", overreaching_class="normal_adaptation"
)
SCENARIO_POST_COMPETITION = ScenarioConfig(
    name="post_competition", description="Post-match recovery.",
    complexity_tier=1, acwr_trajectory="post_match_dip",
    acwr_target_final=(0.6, 0.9), hrv_pattern="post_match_suppression",
    wellness_pattern="post_match_fatigue", load_pattern="minimal_post_match",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="low_if_managed", overreaching_class="normal_adaptation",
    special_parameters={"days_post_match": (1, 4)}
)
SCENARIO_PRESEASON = ScenarioConfig(
    name="preseason_intensification", description="Preseason ramp up.",
    complexity_tier=2, acwr_trajectory="rapid_increase_from_low_base",
    acwr_target_final=(1.3, 1.7), hrv_pattern="mild_suppression_increasing_load",
    wellness_pattern="moderate_fatigue_accumulation", load_pattern="aggressive_ramp_from_zero",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="moderate_to_high", overreaching_class="functional_overreaching",
    special_parameters={"chronic_load_start": "detraining_level", "weeks_since_last_training": (4, 12)}
)
SCENARIO_MONOTONY = ScenarioConfig(
    name="monotony_problem", description="High monotony despite normal ACWR.",
    complexity_tier=2, acwr_trajectory="flat_normal",
    acwr_target_final=(0.9, 1.3), hrv_pattern="slowly_suppressing",
    wellness_pattern="slowly_declining", load_pattern="constant_daily_identical",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="moderate", overreaching_class="functional_overreaching",
    special_parameters={"monotony_index_target": (2.5, 4.0)}
)
SCENARIO_ILLNESS_RETURN = ScenarioConfig(
    name="illness_return", description="Return after illness.",
    complexity_tier=2, acwr_trajectory="undefined_then_spiking",
    acwr_target_final=(1.2, 1.8), hrv_pattern="recovering_but_variable",
    wellness_pattern="improving_but_fragile", load_pattern="graduated_return_from_zero",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="moderate", overreaching_class="normal_adaptation",
    special_parameters={"illness_days": (5, 14), "illness_type": ["upper_respiratory", "gastrointestinal", "flu_like", "general_viral"]}
)
SCENARIO_FIXTURE_CONGESTION = ScenarioConfig(
    name="fixture_congestion", description="Multiple matches per week.",
    complexity_tier=2, acwr_trajectory="match_load_accumulation",
    acwr_target_final=(1.1, 1.5), hrv_pattern="progressive_suppression_matches",
    wellness_pattern="progressive_fatigue_matches", load_pattern="alternating_match_minimal_training",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="moderate_to_high", overreaching_class="functional_overreaching",
    special_parameters={"matches_per_week": (2, 3)}
)
SCENARIO_ALTITUDE = ScenarioConfig(
    name="altitude_camp", description="Training at altitude.",
    complexity_tier=2, acwr_trajectory="moderate_maintained",
    acwr_target_final=(0.9, 1.3), hrv_pattern="altitude_suppression_then_adapting",
    wellness_pattern="altitude_stress_then_improving", load_pattern="reduced_intensity_maintained_volume",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="moderate", overreaching_class="normal_adaptation",
    special_parameters={"altitude_meters": (1800, 3000), "days_at_altitude": (7, 21)}
)
SCENARIO_WELLNESS_CRASH = ScenarioConfig(
    name="wellness_crash_normal_load", description="Normal ACWR but poor wellness.",
    complexity_tier=3, acwr_trajectory="normal_stable",
    acwr_target_final=(0.9, 1.2), hrv_pattern="suppressed_non_load_cause",
    wellness_pattern="severe_crash_non_load", load_pattern="normal_consistent",
    conflicting_signals=True, signal_conflicts=["acwr_normal_but_wellness_crashed", "load_acceptable_but_hrv_suppressed"],
    risk_level="moderate", overreaching_class="normal_adaptation",
    special_parameters={"likely_cause": ["life_stress", "illness_onset", "sleep_disorder", "nutrition_deficit", "travel_disruption"]}
)
SCENARIO_HIGH_ACWR_STABLE = ScenarioConfig(
    name="high_acwr_stable_physiology", description="High ACWR but stable HRV/wellness.",
    complexity_tier=3, acwr_trajectory="elevated_sustained",
    acwr_target_final=(1.5, 1.8), hrv_pattern="stable_despite_high_acwr",
    wellness_pattern="normal_despite_high_acwr", load_pattern="high_but_athlete_tolerating",
    conflicting_signals=True, signal_conflicts=["acwr_danger_but_hrv_normal", "acwr_danger_but_wellness_stable"],
    risk_level="moderate", overreaching_class="functional_overreaching",
    special_parameters={"athlete_level": ["elite", "professional"]}
)
SCENARIO_EARLY_OVERREACHING = ScenarioConfig(
    name="early_overreaching", description="Sustained elevated load for 2-3 weeks.",
    complexity_tier=3, acwr_trajectory="sustained_elevated_multiweek",
    acwr_target_final=(1.3, 1.7), hrv_pattern="progressively_suppressing_multiweek",
    wellness_pattern="progressively_declining_multiweek", load_pattern="consistently_high_multiweek",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="high", overreaching_class="non_functional_overreaching",
    special_parameters={"weeks_elevated": (2, 3)}
)
SCENARIO_OTS = ScenarioConfig(
    name="overtraining_syndrome", description="Overtraining syndrome markers.",
    complexity_tier=3, acwr_trajectory="chronically_elevated_then_declining_performance",
    acwr_target_final=(0.8, 1.3), hrv_pattern="chronically_suppressed",
    wellness_pattern="severely_depressed_all_dimensions", load_pattern="chronically_elevated_then_declining_performance",
    conflicting_signals=True, signal_conflicts=["acwr_now_normal_but_ots_markers_present", "load_reduced_but_performance_still_declining"],
    risk_level="critical", overreaching_class="overtraining_syndrome",
    special_parameters={"weeks_excessive_load": (6, 16)}
)
SCENARIO_YOUTH_PHV = ScenarioConfig(
    name="youth_growth_spurt", description="Youth athlete at PHV.",
    complexity_tier=3, acwr_trajectory="normal_but_phv_risk_elevated",
    acwr_target_final=(0.9, 1.2), hrv_pattern="variable_growth_related",
    wellness_pattern="growing_pains_present", load_pattern="normal_but_requires_phv_modification",
    conflicting_signals=True, signal_conflicts=["acwr_normal_but_phv_elevates_risk", "standard_thresholds_inappropriate_for_phv"],
    risk_level="moderate", overreaching_class="normal_adaptation",
    special_parameters={"phv_status": "at_phv", "growth_rate_cm_per_year": (8, 12), "age_range": (12, 16)}
)
SCENARIO_TRAVEL = ScenarioConfig(
    name="travel_jet_lag", description="International travel with jet lag.",
    complexity_tier=3, acwr_trajectory="reduced_travel_period",
    acwr_target_final=(0.6, 1.0), hrv_pattern="jet_lag_circadian_disruption",
    wellness_pattern="sleep_disrupted_travel", load_pattern="minimal_post_match",
    conflicting_signals=True, signal_conflicts=["low_load_but_poor_hrv_wellness", "non_training_cause_of_suppression"],
    risk_level="moderate", overreaching_class="normal_adaptation",
    special_parameters={"time_zones_crossed": (5, 12), "direction": ["eastward", "westward"], "competition_days_after": (1, 4)}
)
SCENARIO_DOUBLE_SESSIONS = ScenarioConfig(
    name="double_session_accumulation", description="Multiple double-session days.",
    complexity_tier=3, acwr_trajectory="moderate_elevated",
    acwr_target_final=(1.2, 1.6), hrv_pattern="suppressed_inadequate_recovery",
    wellness_pattern="fatigue_accumulating_between_sessions", load_pattern="double_session_days",
    conflicting_signals=True, signal_conflicts=["acwr_moderate_but_recovery_inadequate", "daily_load_ok_but_session_density_problematic"],
    risk_level="moderate_to_high", overreaching_class="functional_overreaching",
    special_parameters={"double_days_per_week": (2, 4), "inter_session_hours": (4, 6)}
)
SCENARIO_HEAT = ScenarioConfig(
    name="heat_acclimatization", description="Training in extreme heat.",
    complexity_tier=3, acwr_trajectory="reduced_intensity_heat",
    acwr_target_final=(0.8, 1.2), hrv_pattern="heat_cardiovascular_suppression",
    wellness_pattern="heat_stress_response", load_pattern="reduced_intensity_maintained_volume",
    conflicting_signals=True, signal_conflicts=["low_load_intensity_but_high_physiological_cost", "environmental_not_training_cause"],
    risk_level="moderate", overreaching_class="normal_adaptation",
    special_parameters={"temperature_celsius": (32, 42), "humidity_percent": (60, 95), "acclimatization_day": (1, 14)}
)
SCENARIO_RECREATIONAL = ScenarioConfig(
    name="recreational_minimal_data", description="Recreational athlete, minimal data.",
    complexity_tier=3, acwr_trajectory="variable_inconsistent_tracking",
    acwr_target_final=(0.8, 1.6), hrv_pattern="not_available",
    wellness_pattern="subjective_only", load_pattern="variable_inconsistent_tracking",
    conflicting_signals=False, signal_conflicts=[],
    risk_level="variable", overreaching_class="normal_adaptation",
    special_parameters={"data_level": 1, "tracking_consistency": "irregular", "monitoring_gaps_days": (1, 5)}
)

ALL_SCENARIOS = {
    "normal_progressive": SCENARIO_NORMAL_PROGRESSIVE,
    "acwr_spike": SCENARIO_ACWR_SPIKE,
    "taper": SCENARIO_TAPER,
    "undertraining": SCENARIO_UNDERTRAINING,
    "post_competition": SCENARIO_POST_COMPETITION,
    "preseason_intensification": SCENARIO_PRESEASON,
    "monotony_problem": SCENARIO_MONOTONY,
    "illness_return": SCENARIO_ILLNESS_RETURN,
    "fixture_congestion": SCENARIO_FIXTURE_CONGESTION,
    "altitude_camp": SCENARIO_ALTITUDE,
    "wellness_crash_normal_load": SCENARIO_WELLNESS_CRASH,
    "high_acwr_stable_physiology": SCENARIO_HIGH_ACWR_STABLE,
    "early_overreaching": SCENARIO_EARLY_OVERREACHING,
    "overtraining_syndrome": SCENARIO_OTS,
    "youth_growth_spurt": SCENARIO_YOUTH_PHV,
    "travel_jet_lag": SCENARIO_TRAVEL,
    "double_session_accumulation": SCENARIO_DOUBLE_SESSIONS,
    "heat_acclimatization": SCENARIO_HEAT,
    "recreational_minimal_data": SCENARIO_RECREATIONAL,
}
TIER_1_SCENARIOS = [k for k,v in ALL_SCENARIOS.items() if v.complexity_tier == 1]
TIER_2_SCENARIOS = [k for k,v in ALL_SCENARIOS.items() if v.complexity_tier == 2]
TIER_3_SCENARIOS = [k for k,v in ALL_SCENARIOS.items() if v.complexity_tier == 3]
