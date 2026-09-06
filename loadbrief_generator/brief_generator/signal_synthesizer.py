# brief_generator/signal_synthesizer.py
# Synthesizes potentially conflicting monitoring signals
# into a coherent clinical interpretation.
# This is the core intelligence of the Tier 3 complex scenarios —
# it understands RELATIONSHIPS between signals, not just individual values.

from typing import Dict, List, Optional

from conflict_rationale import get_rationale

IMPLICATION_PROSE = {
    "athlete_may_be_adapting_well":
        "the athlete may be adapting well despite the mixed signals",
    "non_training_stressor_likely":
        "a non-training stressor is the likely cause",
    "subclinical_fatigue_athlete_unaware":
        "subclinical fatigue the athlete may not yet perceive",
    "psychological_or_contextual_stressor":
        "a psychological or contextual stressor rather than training load",
}

REC_SEVERITY = {
    "no_change": 0,
    "increase_load_gradually": 0,
    "monitor_closely": 1,
    "load_reduction_moderate": 2,
    "load_reduction_significant": 3,
    "rest_and_recovery": 4,
    "medical_review": 5,
}
CLASS_REC_CEILING = {
    "normal_adaptation":           "load_reduction_significant",
    "functional_overreaching":     "rest_and_recovery",
    "non_functional_overreaching": "medical_review",
    "overtraining_syndrome":       "medical_review",
}
class SignalSynthesizer:
    """
    Synthesizes multi-signal athlete monitoring data into a
    structured interpretation object used by TemplateMixer
    to build the output brief.

    Handles three states:
      - Full agreement: all signals point the same direction
      - Partial conflict: one signal disagrees with the others
      - Full conflict: multiple signals disagree
    """

    def __init__(self, config: dict):
        self.acwr_thresholds = config.get("acwr", {})
        self.hrv_thresholds = config.get("hrv", {})

    def synthesize(self,
                   acwr_metrics: Dict,
                   hrv_analysis: Dict,
                   wellness_analysis: Dict,
                   scenario_config,
                   athlete_profile: Dict) -> Dict:
        """
        Main synthesis entry point.
        Returns a structured interpretation dict consumed by
        TemplateMixer.generate_brief().
        """
        # Step 1: Detect signal agreement or conflict
        signal_state = self._detect_signal_state(
            acwr_metrics, hrv_analysis, wellness_analysis, scenario_config
        )

        # Step 2: Route to appropriate interpreter
        agreement = signal_state["agreement"]
        conflicts = signal_state.get("conflicts", [])

        if agreement == "full_agreement":
            interpretation = self._interpret_agreement(
                acwr_metrics, hrv_analysis,
                wellness_analysis, scenario_config
            )
        elif agreement == "partial_conflict":
            interpretation = self._interpret_partial_conflict(
                acwr_metrics, hrv_analysis,
                wellness_analysis, scenario_config,
                conflicts
            )
        else:
            interpretation = self._interpret_full_conflict(
                acwr_metrics, hrv_analysis,
                wellness_analysis, scenario_config,
                conflicts
            )

        # Step 3: Apply athlete-specific context modifiers
        interpretation["athlete_context"] = \
            self._apply_athlete_context(
                interpretation, athlete_profile, scenario_config
            )

        # Step 4: Build the synthesis narrative paragraph
        interpretation["synthesis_narrative"] = \
            self._build_synthesis_narrative(
                interpretation, signal_state,
                scenario_config.risk_level
            )
        declared = getattr(scenario_config, "signal_conflicts", []) or []
        interpretation["scenario_rationale"] = {
            aud: get_rationale(declared, aud)
            for aud in ("athlete", "coach", "sports_scientist")
        }

        # Step 5: Carry forward ground truth labels
        #
        # Capture the SIGNAL-derived recommendation before the declared label
        # overwrites overall_risk — we need both to reconcile them.
        signal_rec = interpretation.get("recommendation_category", "no_change")

        interpretation["overreaching_class"] = \
            scenario_config.overreaching_class
        interpretation["risk_level"] = \
            scenario_config.risk_level
        interpretation["overall_risk"] = \
            scenario_config.risk_level.split("_")[0]

        declared_rec = self._map_risk_to_rec(
            scenario_config.risk_level,
            acwr_metrics.get("zone", "sweet_spot"),
            hrv_analysis.get("consecutive_suppressed_days", 0),
        )
        # Whichever is MORE conservative wins. Declared listed first so it wins
        # ties (max returns the first maximal element).
        rec = max(
            declared_rec, signal_rec,
            key=lambda r: REC_SEVERITY.get(r, 0),
        )
        ceiling = CLASS_REC_CEILING.get(
            scenario_config.overreaching_class, "medical_review")
        if REC_SEVERITY.get(rec, 0) > REC_SEVERITY.get(ceiling, 5):
            rec = ceiling

        interpretation["recommendation_category"] = rec

        return interpretation

    # ── Signal State Detection ────────────────────────────────────────

    def _detect_signal_state(self,
                              acwr_metrics: Dict,
                              hrv_analysis: Dict,
                              wellness_analysis: Dict,
                              scenario_config=None) -> Dict:
        """
        Detect whether monitoring signals agree or conflict.
        Returns a state dict describing any conflicts found.
        """
        conflicts = []

        acwr = acwr_metrics.get("acwr") or 0
        acwr_zone = acwr_metrics.get("zone", "sweet_spot")
        hrv_status = hrv_analysis.get("status", "normal")
        hrv_available = hrv_analysis.get("available", False)
        wellness_status = wellness_analysis.get(
            "composite_status", "good"
        )
        wellness_available = wellness_analysis.get(
            "available", False
        )

        # ── Conflict Type 1 ──────────────────────────────────────────
        # High ACWR but physiology is fine
        # Athlete is tolerating the elevated load well
        if (acwr_zone in ["danger", "extreme"] and
                hrv_available and
                hrv_status in ["normal",
                               "elevated_good_recovery"] and
                wellness_available and
                wellness_status in ["good", "mildly_depressed"]):
            conflicts.append({
                "type": "high_load_healthy_physiology",
                "description": (
                    "Training load is elevated (ACWR in danger zone) "
                    "but physiological markers remain stable — "
                    "the athlete appears to be tolerating the load"
                ),
                "implication": "athlete_may_be_adapting_well",
                "action": "monitor_closely_before_reducing",
                "priority": 2
            })

        # ── Conflict Type 2 ──────────────────────────────────────────
        # Normal ACWR but poor physiology
        # Non-training stressor is likely the cause
        if (acwr_zone in ["sweet_spot", "undertraining"] and
                hrv_available and
                hrv_status in ["significantly_suppressed",
                               "critically_suppressed"] and
                wellness_available and
                wellness_status in ["moderately_depressed",
                                    "severely_depressed"] and
                getattr(scenario_config, "overreaching_class", "")
                    != "overtraining_syndrome"):
            if acwr_zone == "undertraining":
                load_phrase = "Training load is low"
            elif acwr_zone == "caution":
                load_phrase = "Training load is elevated"
            else:
                load_phrase = "Training load is within acceptable range"

            conflicts.append({
                "type": "normal_load_poor_physiology",
                "description": (
                    f"{load_phrase} but "
                    "HRV is suppressed and wellness is depressed — "
                    "a non-training stressor is likely responsible"
                ),
                "implication": "non_training_stressor_likely",
                "action": "investigate_external_stressors",
                "priority": 1  # Highest priority — most important to flag
            })

        # ── Conflict Type 3 ──────────────────────────────────────────
        # HRV suppressed but athlete reporting good wellness
        # Subclinical fatigue — athlete may not yet feel it
        if (hrv_available and
                hrv_status in ["significantly_suppressed",
                               "critically_suppressed"] and
                wellness_available and
                wellness_status in ["good"]):
            conflicts.append({
                "type": "hrv_suppressed_wellness_good",
                "description": (
                    "Objective HRV indicates physiological fatigue "
                    "but the athlete is reporting good subjective state — "
                    "subclinical fatigue the athlete has not yet noticed"
                ),
                "implication": "subclinical_fatigue_athlete_unaware",
                "action": "trust_hrv_over_subjective_report",
                "priority": 3
            })

        # ── Conflict Type 4 ──────────────────────────────────────────
        # HRV normal but wellness poor
        # Psychological or contextual stressor
        if (hrv_available and
                hrv_status in ["normal",
                               "elevated_good_recovery"] and
                wellness_available and
                wellness_status in ["moderately_depressed",
                                    "severely_depressed"]):
            conflicts.append({
                "type": "hrv_normal_wellness_poor",
                "description": (
                    "Physiological recovery (HRV) looks fine but the "
                    "athlete is reporting poor subjective wellbeing — "
                    "a psychological or contextual stressor is likely"
                ),
                "implication": "psychological_or_contextual_stressor",
                "action": "explore_non_physiological_stressors",
                "priority": 3
            })

        # Sort by priority (lower number = higher priority)
        conflicts.sort(key=lambda c: c.get("priority", 99))

        if not conflicts:
            agreement = "full_agreement"
        elif len(conflicts) == 1:
            agreement = "partial_conflict"
        else:
            agreement = "full_conflict"

        return {
            "agreement": agreement,
            "conflicts": conflicts,
            "n_conflicts": len(conflicts)
        }

    # ── Interpreters ─────────────────────────────────────────────────

    def _interpret_agreement(self,
                              acwr_metrics: Dict,
                              hrv_analysis: Dict,
                              wellness_analysis: Dict,
                              scenario_config) -> Dict:
        """
        All signals agree — straightforward interpretation.
        Used for Tier 1 and most Tier 2 scenarios.
        """
        acwr_zone = acwr_metrics.get("zone", "sweet_spot")
        hrv_status = hrv_analysis.get("status", "normal")
        hrv_available = hrv_analysis.get("available", False)
        wellness_status = wellness_analysis.get(
            "composite_status", "good"
        )
        wellness_available = wellness_analysis.get(
            "available", False
        )

        risk_signals = []

        if acwr_zone in ["danger", "extreme"]:
            risk_signals.append("load_elevated")
        elif acwr_zone == "caution":
            risk_signals.append("load_approaching_threshold")
        elif acwr_zone == "undertraining":
            risk_signals.append("load_too_low")

        monotony = acwr_metrics.get("monotony", 0)
        if monotony >= 2.5:
            risk_signals.append("monotony_high")
        elif monotony >= 2.0:
            risk_signals.append("monotony_elevated")

        if hrv_available and hrv_status in [
            "significantly_suppressed", "critically_suppressed"
        ]:
            risk_signals.append("hrv_suppressed")
        elif hrv_available and hrv_status == "moderately_suppressed":
            risk_signals.append("hrv_mildly_suppressed")

        if wellness_available and wellness_status in [
            "moderately_depressed", "severely_depressed"
        ]:
            risk_signals.append("wellness_depressed")
        elif wellness_available and wellness_status == "mildly_depressed":
            risk_signals.append("wellness_mildly_depressed")

        n_risk = len([s for s in risk_signals
                      if "too_low" not in s])

        if n_risk == 0:
            overall_risk = "low"
            primary_message = "all_signals_positive"
        elif n_risk == 1:
            overall_risk = "moderate"
            primary_message = "single_signal_concern"
        elif n_risk == 2:
            overall_risk = "high"
            primary_message = "multiple_signals_concerning"
        else:
            overall_risk = "critical"
            primary_message = "all_signals_critical"

        return {
            "signal_agreement": "full",
            "overall_risk": overall_risk,
            "risk_signals": risk_signals,
            "primary_message": primary_message,
            "confidence": "high",
            "complexity_tier": 1,
            "all_conflicts": [],
            "conditional_recommendations": [],
            "recommendation_category": self._map_risk_to_rec(
                overall_risk,
                acwr_metrics.get("zone", "sweet_spot"),
                hrv_analysis.get(
                    "consecutive_suppressed_days", 0
                )
            )
        }

    def _interpret_partial_conflict(self,
                                     acwr_metrics: Dict,
                                     hrv_analysis: Dict,
                                     wellness_analysis: Dict,
                                     scenario_config,
                                     conflicts: List) -> Dict:
        """
        One signal conflicts — model must explain the discrepancy.
        Used for some Tier 2 and Tier 3 scenarios.
        """
        conflict = conflicts[0]
        ctype = conflict["type"]

        if ctype == "high_load_healthy_physiology":
            resolution = {
                "primary_concern": "acwr_elevated",
                "mitigating_factor": "physiological_tolerance",
                "recommendation_modifier": "monitor_before_reduce",
                "confidence": "moderate",
                "clinical_note": (
                    "Elevated ACWR with stable physiology — "
                    "athlete may be well-adapted. "
                    "Monitor daily before prescribing load reduction."
                )
            }

        elif ctype == "normal_load_poor_physiology":
            resolution = {
                "primary_concern": "non_training_stressor",
                "mitigating_factor": "load_is_not_the_cause",
                "recommendation_modifier": "investigate_external",
                "confidence": "moderate",
                "clinical_note": (
                    "Poor physiological state despite normal load — "
                    "explore non-training stressors before reducing load."
                )
            }

        elif ctype == "hrv_suppressed_wellness_good":
            resolution = {
                "primary_concern": "subclinical_fatigue",
                "mitigating_factor": "athlete_feels_ok",
                "recommendation_modifier": "trust_objective_over_subjective",
                "confidence": "moderate",
                "clinical_note": (
                    "Trust HRV over subjective report — "
                    "objective suppression precedes subjective awareness."
                )
            }

        else:
            resolution = {
                "primary_concern": ctype,
                "mitigating_factor": "signals_partially_disagree",
                "recommendation_modifier": "proceed_with_caution",
                "confidence": "low",
                "clinical_note": "Partial signal conflict — apply clinical judgment."
            }

        return {
            "signal_agreement": "partial",
            "conflict": conflict,
            "resolution": resolution,
            "overall_risk": scenario_config.risk_level.split("_")[0],
            "confidence": resolution["confidence"],
            "complexity_tier": 2,
            "all_conflicts": conflicts,
            "conditional_recommendations": self._build_conditional_recs(
                [conflict], acwr_metrics.get("zone", "sweet_spot")
            ),
            "recommendation_category": self._map_risk_to_rec(
                scenario_config.risk_level.split("_")[0],
                acwr_metrics.get("zone", "sweet_spot"),
                hrv_analysis.get(
                    "consecutive_suppressed_days", 0
                )
            )
        }

    def _interpret_full_conflict(self,
                                  acwr_metrics: Dict,
                                  hrv_analysis: Dict,
                                  wellness_analysis: Dict,
                                  scenario_config,
                                  conflicts: List) -> Dict:
        """
        Multiple signals conflict — highest complexity tier.
        Produces conditional (IF/THEN) recommendations.
        """
        # Already sorted by priority
        primary_conflict = conflicts[0]

        conditional_recs = self._build_conditional_recs(
            conflicts, acwr_metrics.get("zone", "sweet_spot"))

        return {
            "signal_agreement": "conflicting",
            "primary_conflict": primary_conflict,
            "all_conflicts": conflicts,
            "conditional_recommendations": conditional_recs,
            "overall_risk": scenario_config.risk_level.split("_")[0],
            "confidence": "low_due_to_conflict",
            "complexity_tier": 3,
            "clinical_note": (
                "Multiple signal conflicts detected. "
                "Conditional recommendations provided — "
                "clinical judgment and athlete dialogue required."
            ),
            "recommendation_category": self._map_risk_to_rec(
                scenario_config.risk_level.split("_")[0],
                acwr_metrics.get("zone", "sweet_spot"),
                hrv_analysis.get(
                    "consecutive_suppressed_days", 0
                )
            )
        }

    # ── Conditional Recommendation Builder ───────────────────────────

    def _build_conditional_recs(self,
                                  conflicts: List,
                                  acwr_zone: str = "sweet_spot") -> List[Dict]:
        """
        Build IF/THEN conditional recommendations for conflicting signals.
        Each conflict type maps to a pair of conditional recommendations.
        """
        recs = []

        for conflict in conflicts[:2]:  # max 2 conflict types
            ctype = conflict["type"]

            if ctype == "high_load_healthy_physiology":
                recs.append({
                    "condition": "if athlete is elite or well-trained",
                    "action": "maintain current load, monitor daily HRV",
                    "signal": "act immediately if HRV trend turns negative"
                })
                recs.append({
                    "condition": "if athlete is recreational or developing",
                    "action": "reduce load by 15-20% as precaution",
                    "signal": "standard thresholds apply more strictly"
                })

            elif ctype == "normal_load_poor_physiology":
                # "Maintain training load" is correct when the load is
                # genuinely normal, but dangerous when the zone is
                # undertraining — in the depleted OTS presentation the low
                # load IS the symptom, and maintaining it delays recovery.
                if acwr_zone == "undertraining":
                    recs.append({
                        "condition": "if non-training stressor identified",
                        "action": ("address the stressor directly; do not "
                                   "increase load until recovery markers turn"),
                        "signal": "HRV and wellness should respond within 48-72 hours"
                    })
                else:
                    recs.append({
                        "condition": "if non-training stressor identified",
                        "action": "maintain training load, address the stressor directly",
                        "signal": "HRV and wellness should respond within 48-72 hours"
                    })
                recs.append({
                    "condition": "if no stressor identified after investigation",
                    "action": "reduce load 20% as precaution",
                    "signal": "reassess in 3 days — if no improvement, seek medical review"
                })

            elif ctype == "hrv_suppressed_wellness_good":
                # Reducing intensity is right when load is normal or high, but
                # counterproductive at an undertraining ratio — the athlete is
                # already under-loaded, and cutting further deepens it. 1,721
                # briefs recommended a reduction at ACWR ~0.45.
                if acwr_zone == "undertraining":
                    recs.append({
                        "condition": "given subclinical fatigue pattern",
                        "action": ("hold load steady and investigate recovery "
                                   "quality before progressing; do not reduce "
                                   "further"),
                        "signal": "monitor whether subjective state deteriorates to match HRV"
                    })
                else:
                    recs.append({
                        "condition": "given subclinical fatigue pattern",
                        "action": "reduce next session intensity to RPE ≤ 6",
                        "signal": "monitor whether subjective state deteriorates to match HRV"
                    })

            elif ctype == "hrv_normal_wellness_poor":
                recs.append({
                    "condition": "given psychological stressor pattern",
                    "action": "check in with athlete about non-training stressors",
                    "signal": "training load modification alone is unlikely to help"
                })

        return recs

    # ── Context and Narrative Builders ───────────────────────────────

    def _apply_athlete_context(self,
                                interpretation: Dict,
                                athlete_profile: Dict,
                                scenario_config) -> Dict:
        """
        Modify interpretation based on athlete-specific context.
        Elite athletes tolerate different thresholds.
        Youth athletes require conservative modifications.
        """
        modifiers = []
        level = athlete_profile.get("level", "recreational")
        age = athlete_profile.get("age", 25)
        special = scenario_config.special_parameters

        if level in ["professional", "elite"]:
            modifiers.append({
                "modifier": "elite_tolerance",
                "effect": "thresholds_adjusted_upward",
                "note": "Elite athletes commonly tolerate higher ACWR"
            })

        if special.get("phv_status") == "at_phv" or age < 16:
            modifiers.append({
                "modifier": "youth_phv_risk",
                "effect": "thresholds_adjusted_downward",
                "note": (
                    "PHV status elevates injury risk at loads "
                    "that would be safe for adult athletes"
                )
            })

        for env_key in ["altitude_meters",
                        "temperature_celsius",
                        "time_zones_crossed"]:
            if env_key in special:
                modifiers.append({
                    "modifier": f"environmental_{env_key}",
                    "effect": "signals_partially_explained_by_environment",
                    "value": special[env_key]
                })

        return {
            "modifiers": modifiers,
            "modified": len(modifiers) > 0
        }

    def _build_synthesis_narrative(self,
                                    interpretation: Dict,
                                    signal_state: Dict,
                                    risk_level: str = "low") -> str:
        """
        Builds the signal integration narrative paragraph.
        This becomes the SIGNAL INTEGRATION section of the brief.
        """
        agreement = signal_state["agreement"]
        conflicts = signal_state.get("conflicts", [])
        risk_signals = interpretation.get("risk_signals", [])

        if agreement == "full_agreement":
            n_risk = len([s for s in risk_signals
                          if "too_low" not in s])

            if n_risk == 0:
                # risk_signals counts only threshold crossings, so it can be
                # empty while the scenario still carries elevated risk. An
                # absolute all-clear then contradicts the brief's own risk
                # header — the defect this guard prevents.
                if (risk_level or "").split("_")[0] in (
                        "moderate", "high", "critical"):
                    return (
                        "No individual monitoring signal has crossed its "
                        "threshold, but the overall picture warrants "
                        "continued attention rather than reassurance."
                    )
                # Scoped to what was MONITORED rather than an absolute
                # completeness claim — under reduced data levels the brief
                # cannot honestly assert that every signal is normal.
                return (
                    "No monitored signal has crossed its threshold. Training "
                    "load appears well tolerated with adequate recovery."
                )
            elif n_risk == 1:
                signal = risk_signals[0].replace("_", " ")
                return (
                    f"Monitoring signals are largely positive with one "
                    f"area of concern: {signal}. "
                    f"Other signals remain within acceptable ranges."
                )
            elif n_risk == 2:
                # Name the signals that ACTUALLY flagged. The previous version
                # hardcoded "load and physiological recovery", false whenever
                # the two flagged signals were e.g. HRV + wellness (load fine).
                flagged = [s for s in risk_signals if "too_low" not in s]
                label = {
                    "load_elevated": "training load",
                    "load_approaching_threshold": "training load",
                    "hrv_suppressed": "physiological recovery",
                    "hrv_mildly_suppressed": "physiological recovery",
                    "wellness_depressed": "subjective wellness",
                    "wellness_mildly_depressed": "subjective wellness",
                    "monotony_high": "training monotony",
                    "monotony_elevated": "training monotony",
                }
                named = []
                for s in flagged:
                    n = label.get(s, s.replace("_", " "))
                    if n not in named:
                        named.append(n)
                pair = (" and ".join(named) if len(named) == 2
                        else "two monitoring signals")
                return (
                    f"Two monitoring signals are simultaneously indicating "
                    f"elevated fatigue — {pair} are both showing stress. "
                    f"Combined signal elevation is more concerning than "
                    f"either signal would be in isolation."
                )
            else:
                return (
                    "All primary monitoring signals are converging to "
                    "indicate significant fatigue accumulation. "
                    "Training load, physiological readiness, and "
                    "subjective wellness are simultaneously flagging — "
                    "this pattern is consistent with non-functional "
                    "overreaching and requires immediate attention."
                )

        elif agreement == "partial_conflict":
            conflict = conflicts[0]
            return (
                f"Monitoring signals show a notable discrepancy: "
                f"{conflict['description']}. "
                f"This pattern suggests "
                f"{IMPLICATION_PROSE.get(conflict['implication'], conflict['implication'].replace('_', ' '))}. "
                f"Recommended interpretation: "
                f"{conflict['action'].replace('_', ' ')}."
            )

        else:  # full_conflict
            primary = conflicts[0]
            n = len(conflicts)
            return (
                f"Monitoring signals are conflicting across "
                f"{n} dimensions. "
                f"The most clinically significant discrepancy: "
                f"{primary['description']}. "
                f"This complexity warrants conditional recommendations "
                f"rather than a single prescribed response. "
                f"Clinical judgment and direct athlete dialogue are "
                f"essential before acting on this data."
            )

    # ── Utility ──────────────────────────────────────────────────────

    def _map_risk_to_rec(self, risk: str,
                          acwr_zone: str,
                          hrv_days: int) -> str:
        """Map risk level to recommendation category"""
        if "critical" in risk:
            return "medical_review"
        elif "high" in risk:
            if hrv_days >= 7:
                return "rest_and_recovery"
            return "load_reduction_significant"
        elif "moderate" in risk:
            if hrv_days >= 3:
                return "load_reduction_moderate"
            return "monitor_closely"
        else:
            if acwr_zone == "undertraining":
                return "increase_load_gradually"
            return "no_change"
