# brief_generator/template_mixer.py
# Builds the complete output brief from a synthesis interpretation.
# Produces linguistically varied text across all 4 audience types.
# Without variation, all 50,000 output briefs would sound identical.

import random
from typing import Dict, List


# ── Risk Level Openers ────────────────────────────────────────────────
RISK_OPENERS = {
    "low": [
        "Monitoring data for this period looks positive.",
        "Current load management is on track.",
        "Training signals are within healthy parameters.",
        "No significant concerns identified in this monitoring period.",
        "The athlete appears to be tolerating training well.",
        "Recovery and load balance are functioning effectively.",
        "All indicators suggest this training block is progressing well.",
        "Current monitoring status: no action required.",
        "Load management appears well-calibrated this period.",
        "The data reflects a well-managed training phase."
    ],
    "moderate": [
        "Some monitoring signals warrant attention.",
        "Current data suggests emerging fatigue — proactive management recommended.",
        "There are early indicators of load accumulation worth addressing.",
        "Monitoring flags a moderate concern requiring closer tracking.",
        "The picture is mixed — most signals acceptable but one area needs attention.",
        "This period shows manageable fatigue that should be addressed proactively.",
        "Monitoring data suggests the athlete is approaching their tolerance threshold.",
        "Caution is warranted — early intervention is easier than late correction.",
        "The data suggests a moderate load-recovery imbalance developing.",
        "Several signals are trending in a direction that warrants adjustment."
    ],
    "high": [
        "Monitoring data indicates elevated injury and overreaching risk.",
        "Multiple signals are converging to indicate significant fatigue accumulation.",
        "Current load management requires immediate attention.",
        "The athlete is showing clear signs of non-functional overreaching.",
        "Training load has exceeded recovery capacity — intervention required.",
        "This monitoring period shows a pattern consistent with high injury risk.",
        "Urgent load management action is recommended based on current data.",
        "The combination of signals present indicates a high-risk state.",
        "Monitoring data is clearly indicating overreaching — act now.",
        "All primary risk indicators are elevated simultaneously."
    ],
    "critical": [
        "Critical: Monitoring data indicates overtraining syndrome markers.",
        "All monitoring signals are at critical levels — immediate intervention required.",
        "This athlete requires urgent clinical assessment.",
        "The data is consistent with overtraining syndrome — training must be suspended.",
        "CRITICAL ALERT: Multi-system fatigue markers are at severe levels.",
        "Immediate and significant load reduction is required.",
        "Clinical review is strongly recommended alongside immediate training halt.",
        "All signals are in the danger zone simultaneously.",
        "Overtraining syndrome indicators present — physician review required.",
        "The severity of current signals requires immediate medical and coaching intervention."
    ],
    "variable": [
        "Monitoring data is available with limited signals for this athlete.",
        "Assessment based on available data — full monitoring would improve confidence.",
        "Partial monitoring data available — interpretation carries higher uncertainty."
    ]
}

# ── Overreaching Classification Language ──────────────────────────────
OC_LANGUAGE = {
    "normal_adaptation": {
        "athlete":          "Your body is handling training well.",
        "coach":            "Normal adaptation — no overreaching indicators.",
        "sports_scientist": "Normal adaptation. No functional or non-functional overreaching markers present."
    },
    "undertraining": {
        "athlete":          "Your training load may be too low — you could be losing fitness.",
        "coach":            "Undertraining risk — load may be insufficient to drive adaptation.",
        "sports_scientist": "Undertraining state indicated (ACWR below 0.8). Chronic adaptation at risk."
    },
    "functional_overreaching": {
        "athlete":          "You are showing early overtraining signs. A few easy days will resolve this.",
        "coach":            "Functional overreaching — short recovery period (3-7 days) will restore performance.",
        "sports_scientist": "Functional overreaching (Meeusen et al. 2013). Expected recovery within 3-7 days with appropriate load reduction."
    },
    "non_functional_overreaching": {
        "athlete":          "Your body is showing overtraining warning signs that need 2-6 weeks of reduced training.",
        "coach":            "Non-functional overreaching — recovery requires 2-6 weeks of significantly reduced load.",
        "sports_scientist": "Non-functional overreaching (Meeusen et al. 2013). Recovery timeline 2-6 weeks. Performance decline expected without intervention."
    },
    "overtraining_syndrome": {
        "athlete":          "Your body is showing severe overtraining signs. You need extended rest and medical support.",
        "coach":            "Overtraining syndrome suspected — physician review required. Training must be suspended.",
        "sports_scientist": "Overtraining syndrome markers present (Meeusen et al. 2013). Immediate training suspension and clinical assessment required. Recovery timeline months."
    }
}

# ── Recommendation Templates ──────────────────────────────────────────
RECOMMENDATION_TEMPLATES = {
    "no_change": [
        "Continue current training plan as prescribed.",
        "No modifications to training are indicated.",
        "Maintain current load — all signals support continued progression.",
        "Proceed with the planned training block without modification."
    ],
    "increase_load_gradually": [
        "Consider a gradual 5-10% load increase to avoid detraining.",
        "Training load can be progressively increased — chronic base is established.",
        "A conservative load increase of 5-10% per week is appropriate."
    ],
    "monitor_closely": [
        "Increase monitoring frequency — daily HRV and wellness check-ins recommended.",
        "No load change required but daily tracking should be maintained.",
        "Continue training with heightened monitoring attention.",
        "Maintain current load while monitoring daily for deterioration."
    ],
    "load_reduction_moderate": [
        "Reduce session volume by 20-30% over the next 3-4 days while maintaining intensity.",
        "A moderate volume reduction (20-30%) is recommended for the next training block.",
        "Pull back total training volume by approximately 25% while preserving movement quality.",
        "Reduce load by 20-30% for 3-5 days, then reassess based on HRV response."
    ],
    "load_reduction_significant": [
        "Reduce overall training load by 40-50% — replace high-intensity sessions with aerobic threshold work at RPE ≤ 6.",
        "Significant load reduction required: cut volume by 40-50%, eliminate all high-intensity work for 5-7 days.",
        "Drop training volume by half and restrict intensity to conversational aerobic work for one week minimum.",
        "Immediate 40-50% load reduction. No sessions above RPE 6 for at least 5 days."
    ],
    "rest_and_recovery": [
        "Complete rest or active recovery only (light walking, swimming) for 3-5 days minimum.",
        "Training should be suspended — active recovery protocols only until markers normalize.",
        "Full rest recommended. No structured training until HRV and wellness return to individual baseline.",
        "3-5 days complete rest followed by gradual return if markers normalise."
    ],
    "medical_review": [
        "Physician review is strongly recommended alongside training suspension.",
        "Clinical assessment is required before training resumes.",
        "Refer to sports medicine for evaluation — current markers are consistent with overtraining syndrome.",
        "Immediate medical review required. Training suspended pending clinical assessment."
    ],
    "investigate_external_stressors": [
        "Investigate non-training stressors before modifying load — sleep, nutrition, life stress, and illness should all be explored.",
        "Check in with the athlete about life stressors, sleep quality, and nutrition before reducing training load.",
        "Load modification alone is unlikely to resolve this pattern. Identify and address the underlying stressor first."
    ]
}

# ── Escalation Triggers ───────────────────────────────────────────────
ESCALATION_TRIGGERS = {
    "athlete": [
        "SEEK SUPPORT if you feel exhausted after rest, your mood is consistently low, or performance keeps dropping despite reduced training.",
        "Talk to your coach or doctor if things do not improve after a few easy days.",
        "Flag to your support team if you notice persistent fatigue, mood changes, or performance decline."
    ],
    "coach": [
        "Escalate to physician review if: HRV suppression persists beyond 7 days; performance declines despite load reduction; athlete reports inability to complete prescribed sessions.",
        "Physician referral warranted if markers do not improve within 5-7 days of load reduction.",
        "Escalation criteria: persistent multi-system suppression, performance decline, or athlete distress."
    ],
    "sports_scientist": [
        "Escalation criteria: HRV suppression >7 consecutive days unresponsive to load reduction; performance decline on standardised testing; neuroendocrine marker elevation where available. Refer to sports medicine physician.",
        "Clinical referral warranted if: non-functional overreaching markers persist >2 weeks; performance loss exceeds 5% on objective measures; multi-system fatigue markers are severe.",
        "Physician review indicated if overtraining syndrome criteria (Meeusen et al. 2013) are met or approached."
    ]
}

# ── Monitoring Priority Templates ────────────────────────────────────
MONITORING_PRIORITIES = {
    "low": [
        "Standard monitoring protocol — no changes required.",
        "Continue current monitoring schedule.",
        "No additional monitoring interventions needed at this time."
    ],
    "moderate": [
        "Daily wellness check-in before sessions. HRV on all training days.",
        "Increase monitoring frequency: daily HRV + pre-session wellness questionnaire.",
        "Daily morning HRV and wellness check-in. Reassess in 3-5 days."
    ],
    "high": [
        "Daily morning HRV measurement. Wellness questionnaire before every session. Session RPE immediately post-session. Reassess in 3 days.",
        "Intensive monitoring protocol: daily HRV + daily wellness + post-session RPE. Review in 48-72 hours.",
        "Maximum monitoring frequency. Daily HRV, daily wellness, post-session RPE. Daily coach check-in with athlete."
    ],
    "critical": [
        "Clinical monitoring required. Daily physician or practitioner assessment alongside training suspension.",
        "Medical monitoring protocol. No training until cleared by physician.",
        "Suspend standard monitoring protocol. Immediate clinical assessment takes priority."
    ]
}


class TemplateMixer:
    """
    Builds the complete output brief from a synthesis interpretation.
    Uses template libraries with random selection to produce
    natural language variation across all training examples.
    """

    def __init__(self, audience_profiles: dict):
        self.audience_profiles = audience_profiles

    def generate_brief(self,
                       synthesis: Dict,
                       acwr_metrics: Dict,
                       hrv_analysis: Dict,
                       wellness_analysis: Dict,
                       athlete_profile: Dict,
                       audience: str,
                       scenario_config) -> str:
        """
        Main entry point — generates a complete output brief
        with natural language variation for the specified audience.
        """
        risk = synthesis.get("overall_risk", "low")

        # Clean risk level for template lookup
        risk_key = risk.split("_")[0] if "_" in risk else risk
        if risk_key not in RISK_OPENERS:
            risk_key = "moderate"

        sections = []

        # 1. Risk level declaration + opener
        sections.append(
            self._build_risk_opener(risk_key, audience)
        )

        # 2. Load analysis
        sections.append(
            self._build_load_section(
                acwr_metrics, audience, synthesis
            )
        )

        # 3. HRV section (if available)
        if hrv_analysis.get("available", False):
            sections.append(
                self._build_hrv_section(hrv_analysis, audience)
            )

        # 4. Wellness section (if available)
        if wellness_analysis.get("available", False):
            sections.append(
                self._build_wellness_section(
                    wellness_analysis, audience
                )
            )

        # 5. Signal integration narrative
        synthesis_narrative = synthesis.get(
            "synthesis_narrative", ""
        )
        if synthesis_narrative:
            sections.append(
                f"SIGNAL INTEGRATION:\n{synthesis_narrative}"
            )

        # 6. Overreaching classification
        sections.append(
            self._build_classification_section(
                synthesis, audience
            )
        )

        # 7. Conflict explanation (Tier 3 only)
        conflicts = synthesis.get("all_conflicts", [])
        if conflicts:
            sections.append(
                self._build_conflict_section(
                    conflicts, synthesis, audience
                )
            )

        # 8. Recommendations
        sections.append(
            self._build_recommendations_section(
                synthesis, acwr_metrics,
                hrv_analysis, audience
            )
        )

        # 9. Monitoring priorities
        sections.append(
            self._build_monitoring_section(
                risk_key, synthesis, audience
            )
        )

        # 10. Escalation trigger
        sections.append(
            self._build_escalation_section(audience, risk_key)
        )

        # Join non-empty sections
        brief = "\n\n".join(
            s for s in sections if s and s.strip()
        )

        return brief.strip()

    # ── Section Builders ─────────────────────────────────────────────

    def _build_risk_opener(self, risk_key: str,
                            audience: str) -> str:
        """Risk level declaration with varied opener"""
        opener = random.choice(
            RISK_OPENERS.get(risk_key, RISK_OPENERS["moderate"])
        )

        if audience == "athlete":
            return f"YOUR STATUS: {risk_key.upper()}\n\n{opener}"
        elif audience == "sports_scientist":
            return (
                f"OVERALL RISK CLASSIFICATION: {risk_key.upper()}\n\n"
                f"{opener}"
            )
        else:
            return f"RISK LEVEL: {risk_key.upper()}\n\n{opener}"

    def _build_load_section(self, acwr_metrics: Dict,
                             audience: str,
                             synthesis: Dict) -> str:
        """Build training load section"""
        acwr = acwr_metrics.get("acwr")
        zone = acwr_metrics.get("zone", "sweet_spot")
        acute = acwr_metrics.get("acute_load", 0)
        chronic = acwr_metrics.get("chronic_load", 0)
        monotony = acwr_metrics.get("monotony", 0)

        if acwr is None:
            return (
                "TRAINING LOAD:\n"
                "Insufficient data for full load analysis."
            )

        # Athlete gets plain language
        if audience == "athlete":
            zone_plain = {
                "undertraining": f"Your training load is low ({acwr:.2f} ratio). You may be losing fitness.",
                "sweet_spot": f"Your training load is well-balanced ({acwr:.2f} ratio). Good work.",
                "caution": f"Your training load is getting high ({acwr:.2f} ratio). Worth watching.",
                "danger": f"Your training load spiked recently ({acwr:.2f} ratio). Your body is under significant stress.",
                "extreme": f"Your training load is dangerously high ({acwr:.2f} ratio). You need to reduce training immediately."
            }
            load_text = zone_plain.get(
                zone,
                f"Training load ratio: {acwr:.2f}."
            )
        else:
            # Technical version with raw values
            zone_descriptions = {
                "undertraining": f"ACWR of {acwr:.3f} indicates undertraining (below 0.8 threshold).",
                "sweet_spot": f"ACWR of {acwr:.3f} is within the optimal training range.",
                "caution": f"ACWR of {acwr:.3f} is in the caution zone — approaching upper threshold.",
                "danger": f"ACWR of {acwr:.3f} is in the danger zone (above 1.5) — injury risk elevated.",
                "extreme": f"ACWR of {acwr:.3f} is at an extreme level — very high injury risk."
            }
            load_text = zone_descriptions.get(
                zone,
                f"ACWR: {acwr:.3f}."
            )

            if audience in ["coach", "sports_scientist"]:
                load_text += (
                    f" Acute load: {acute:.0f} AU | "
                    f"Chronic load: {chronic:.0f} AU."
                )

            # Monotony note for sports scientists
            if (audience == "sports_scientist" and
                    monotony is not None and monotony > 0):
                if monotony > 2.0:
                    load_text += (
                        f" Training monotony: {monotony:.2f} "
                        f"({'HIGH RISK' if monotony > 3.0 else 'elevated'})."
                    )
                else:
                    load_text += (
                        f" Training monotony: {monotony:.2f} (acceptable)."
                    )

        return f"TRAINING LOAD:\n{load_text}"

    def _build_hrv_section(self, hrv_analysis: Dict,
                            audience: str) -> str:
        """Build HRV section"""
        status = hrv_analysis.get("status", "normal")
        current_delta = hrv_analysis.get(
            "current_vs_baseline_ms", 0
        )
        mean_delta = hrv_analysis.get(
            "7day_mean_vs_baseline_ms", 0
        )
        baseline = hrv_analysis.get("baseline", 65)
        days = hrv_analysis.get(
            "consecutive_suppressed_days", 0
        )
        recent = hrv_analysis.get("recent_values", [])
        trend = hrv_analysis.get("trend_direction", "stable")

        current_val = baseline + current_delta
        delta_abs = abs(mean_delta)

        if audience == "athlete":
            status_plain = {
                "normal": "Your recovery score is normal — good to go.",
                "elevated_good_recovery": "Your recovery score is above normal — you are well-rested.",
                "moderately_suppressed": "Your recovery score is slightly low — some fatigue building.",
                "significantly_suppressed": f"Your recovery score has been low for {days} days — your body needs more rest.",
                "critically_suppressed": f"Your recovery score has been very low for {days} days in a row — this is a clear signal to rest.",
                "data_unavailable": "Recovery score data not available."
            }
            hrv_text = status_plain.get(
                status, "Recovery status unclear."
            )
        else:
            # Technical version
            status_text = {
                "normal": (
                    f"HRV normal: {current_val:.1f}ms "
                    f"({current_delta:+.1f}ms vs "
                    f"{baseline:.1f}ms baseline)."
                ),
                "elevated_good_recovery": (
                    f"HRV elevated: {current_val:.1f}ms "
                    f"(+{delta_abs:.1f}ms above baseline) — "
                    f"excellent recovery signal."
                ),
                "moderately_suppressed": (
                    f"HRV moderately suppressed: {current_val:.1f}ms "
                    f"({delta_abs:.1f}ms below {baseline:.1f}ms baseline)."
                ),
                "significantly_suppressed": (
                    f"HRV significantly suppressed: {current_val:.1f}ms "
                    f"({delta_abs:.1f}ms below baseline, "
                    f"{days} consecutive days)."
                ),
                "critically_suppressed": (
                    f"HRV critically suppressed: {current_val:.1f}ms "
                    f"({delta_abs:.1f}ms below {baseline:.1f}ms baseline "
                    f"for {days} consecutive days)."
                ),
                "data_unavailable": "HRV data unavailable for this period."
            }
            hrv_text = status_text.get(
                status, f"HRV status: {status}."
            )

            if trend != "stable" and audience == "sports_scientist":
                hrv_text += f" 7-day trend: {trend}."

        return f"PHYSIOLOGICAL READINESS (HRV):\n{hrv_text}"

    def _build_wellness_section(self, wellness_analysis: Dict,
                                  audience: str) -> str:
        """Build wellness section"""
        composite = wellness_analysis.get(
            "composite_status", "good"
        )
        dims = wellness_analysis.get("dimensions", {})
        red = wellness_analysis.get("red_flags", 0)
        amber = wellness_analysis.get("amber_flags", 0)

        problem_dims = [
            (dim, data) for dim, data in dims.items()
            if data.get("zone") in ["red", "amber"]
        ]

        if audience == "athlete":
            if not problem_dims:
                wellness_text = "You are feeling good — sleep, energy, soreness, and mood all look normal."
            elif len(problem_dims) == 1:
                dim, data = problem_dims[0]
                dim_plain = dim.replace("_", " ")
                wellness_text = (
                    f"One concern: your {dim_plain} is below normal "
                    f"and trending {data.get('trend', 'stable')}."
                )
            else:
                dims_plain = " and ".join([
                    d[0].replace("_", " ")
                    for d in problem_dims[:3]
                ])
                wellness_text = (
                    f"Multiple wellness areas are flagged: {dims_plain}. "
                    f"Overall you are feeling {composite.replace('_', ' ')}."
                )
        else:
            if not problem_dims:
                wellness_text = (
                    "All wellness dimensions within normal range. "
                    "Composite status: good."
                )
            else:
                dim_list = ", ".join([
                    f"{d[0].replace('_',' ')} "
                    f"({d[1].get('zone','amber')}, "
                    f"{d[1].get('trend','stable')})"
                    for d in problem_dims
                ])
                wellness_text = (
                    f"Composite status: "
                    f"{composite.replace('_', ' ')}. "
                    f"Flagged dimensions: {dim_list}. "
                    f"Red flags: {red}, Amber flags: {amber}."
                )

        return f"SUBJECTIVE WELLNESS:\n{wellness_text}"

    def _build_classification_section(self,
                                        synthesis: Dict,
                                        audience: str) -> str:
        """Build overreaching classification section"""
        oc = synthesis.get(
            "overreaching_class", "normal_adaptation"
        )
        confidence = synthesis.get("confidence", "high")

        oc_text = OC_LANGUAGE.get(oc, {}).get(
            audience,
            oc.replace("_", " ").title()
        )

        conf_note = ""
        if confidence not in ["high"] and \
                audience == "sports_scientist":
            conf_note = f" (confidence: {confidence})"

        if audience == "athlete":
            return f"CURRENT STATUS:\n{oc_text}"
        elif audience == "sports_scientist":
            return (
                f"OVERREACHING CLASSIFICATION:\n"
                f"{oc_text}{conf_note}"
            )
        else:
            return (
                f"CLINICAL CLASSIFICATION:\n{oc_text}{conf_note}"
            )

    def _build_conflict_section(self,
                                  conflicts: List,
                                  synthesis: Dict,
                                  audience: str) -> str:
        """Build signal conflict explanation (Tier 3 only)"""
        if not conflicts or audience == "athlete":
            return ""

        primary = conflicts[0]
        n = len(conflicts)

        if audience == "coach":
            conflict_text = (
                f"NOTE — Signal Conflict Detected: "
                f"{primary['description']}. "
                f"This requires clinical judgment rather than "
                f"automatic protocol application."
            )
        else:  # sports_scientist
            all_desc = "; ".join([
                c["description"] for c in conflicts
            ])
            conflict_text = (
                f"SIGNAL CONFLICTS ({n} detected):\n"
                f"{all_desc}\n"
                f"Recommended approach: "
                f"{primary['action'].replace('_', ' ')}."
            )

        return f"SIGNAL CONFLICT ANALYSIS:\n{conflict_text}"

    def _build_recommendations_section(self,
                                         synthesis: Dict,
                                         acwr_metrics: Dict,
                                         hrv_analysis: Dict,
                                         audience: str) -> str:
        """Build recommendations with conditional logic for Tier 3"""
        rec_category = synthesis.get(
            "recommendation_category", "no_change"
        )
        conditional_recs = synthesis.get(
            "conditional_recommendations", []
        )
        conflicts = synthesis.get("all_conflicts", [])

        recs = []

        # Primary recommendation
        if conflicts and conditional_recs:
            # Tier 3 — use conditional recommendations
            for cr in conditional_recs[:2]:
                recs.append(
                    f"• IF {cr['condition']}: "
                    f"{cr['action']} — "
                    f"signal to watch: {cr['signal']}"
                )
        else:
            # Tier 1/2 — use standard recommendation
            templates = RECOMMENDATION_TEMPLATES.get(
                rec_category,
                RECOMMENDATION_TEMPLATES["no_change"]
            )
            recs.append(f"• {random.choice(templates)}")

        # Add context-specific note for external stressors
        if any(
            c.get("type") == "normal_load_poor_physiology"
            for c in conflicts
        ):
            recs.append(
                "• Athlete conversation recommended before "
                "any load modification — explore sleep, "
                "nutrition, life stress, and illness."
            )

        return f"RECOMMENDATIONS:\n" + "\n".join(recs)

    def _build_monitoring_section(self, risk_key: str,
                                    synthesis: Dict,
                                    audience: str) -> str:
        """Build monitoring priorities section"""
        if audience == "athlete":
            return ""  # Athletes don't need monitoring protocol

        priorities = MONITORING_PRIORITIES.get(
            risk_key,
            MONITORING_PRIORITIES["moderate"]
        )
        priority_text = random.choice(priorities)

        # Add conflict-specific monitoring note
        conflicts = synthesis.get("all_conflicts", [])
        if conflicts:
            priority_text += (
                " Athlete interview recommended to identify "
                "potential non-training stressors."
            )

        return f"MONITORING PRIORITIES:\n{priority_text}"

    def _build_escalation_section(self,
                                    audience: str,
                                    risk_key: str) -> str:
        """Build escalation triggers section"""
        triggers = ESCALATION_TRIGGERS.get(
            audience,
            ESCALATION_TRIGGERS["coach"]
        )
        trigger_text = random.choice(triggers)

        # Only include for moderate+ risk
        if risk_key == "low" and audience == "coach":
            return ""

        return f"ESCALATION TRIGGERS:\n{trigger_text}"
