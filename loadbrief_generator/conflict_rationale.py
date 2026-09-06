#!/usr/bin/env python3
# conflict_rationale.py
#
# ROOT CAUSE THIS FIXES
# ---------------------
# signal_synthesizer._detect_signal_state() derives conflicts from the NUMBERS
# (HRV vs wellness vs ACWR). That works for numerically-visible conflicts such
# as "HRV suppressed but wellness good" -- which is why illness_return briefs
# already contain good conflict prose.
#
# But scenario-CONTEXTUAL overrides are invisible in the numbers. Nothing in
# an HRV/wellness/ACWR comparison can reveal that the suppression is caused by
# heat, altitude, jet lag, or adolescent growth. That knowledge exists only in
# ScenarioConfig.signal_conflicts -- which is declared for every scenario in
# scenarios.py and READ NOWHERE in the codebase.
#
# Result: override scenarios emit a benign label over adverse-signal prose with
# no explanation, because the explanation was never plumbed through. This maps
# each declared tag to per-audience rationale prose so the brief can say WHY.
#
# COUPLING WARNING
# ----------------
# RATIONALE_PATTERNS_FOR_CHECKER at the bottom must match the phrases emitted
# here. If they drift apart, consistency_check.py will flag every correctly
# rewritten override brief as "missing rationale". Change them together.

from typing import Dict, List, Optional


# tag -> {audience: prose}. Tags come verbatim from scenarios.py.
CONFLICT_RATIONALE: Dict[str, Dict[str, str]] = {

    # ── heat_acclimatization ──────────────────────────────────────────
    "environmental_not_training_cause": {
        "athlete": ("Your recovery numbers look low, but this is an expected "
                    "response to training in the heat rather than a sign of "
                    "overreaching. It typically settles as you acclimatise."),
        "coach": ("HRV suppression here is an expected environmental response "
                  "to heat exposure rather than training-induced fatigue. "
                  "Interpret against heat-acclimatisation norms, not standard "
                  "load thresholds."),
        "sports_scientist": ("Cardiovascular drift and HRV suppression are "
                             "attributable to thermoregulatory load rather "
                             "than training stress. Expected environmental "
                             "response during heat acclimatisation; standard "
                             "overreaching criteria do not apply unmodified."),
    },
    "low_load_intensity_but_high_physiological_cost": {
        "athlete": ("Your training numbers may look manageable, but the heat makes each session cost your body more."),
        "coach": ("Training load metrics understate the true strain in these conditions — physiological cost is elevated independent of workload."),
        "sports_scientist": ("External load underrepresents internal load: "
                             "thermoregulatory demand elevates physiological "
                             "cost independent of mechanical work."),
    },

    # ── youth_growth_spurt ────────────────────────────────────────────
    "acwr_normal_but_phv_elevates_risk": {
        "athlete": ("Your training load looks fine on paper, but growing "
                    "quickly puts extra strain on your body right now."),
        "coach": ("ACWR is within range, but peak height velocity elevates "
                  "injury risk independently of workload. This is "
                  "developmentally expected rather than a training error."),
        "sports_scientist": ("Elevated risk during PHV is growth-related "
                             "rather than load-driven; ACWR alone "
                             "underestimates risk in this maturation window."),
    },
    "standard_thresholds_inappropriate_for_phv": {
        "athlete": ("The usual training guidelines don't fit while you're "
                    "growing this fast."),
        "coach": ("Standard ACWR thresholds are not appropriate during peak "
                  "height velocity; apply maturation-adjusted limits."),
        "sports_scientist": ("Chronological thresholds are invalid during "
                             "PHV; maturity-adjusted criteria required."),
    },

    # ── overtraining_syndrome ─────────────────────────────────────────
    # The register defect lives here: the athlete layer framed low load as
    # "you may be losing fitness" under a CRITICAL call. Low load is a
    # SYMPTOM of the depleted state, not a detraining concern.
    "acwr_now_normal_but_ots_markers_present": {
        "athlete": ("Your training load is low right now because your body "
                    "cannot sustain more - this is part of the problem, not "
                    "a fitness concern. Chronic fatigue markers remain "
                    "elevated despite the reduced load."),
        "coach": ("Load has normalised but overtraining markers persist. The "
                  "low ACWR reflects accumulated chronic fatigue rather than "
                  "detraining, and does not indicate recovery."),
        "sports_scientist": ("ACWR normalisation does not indicate recovery: "
                             "chronic multi-system markers persist despite "
                             "reduced current load, consistent with the "
                             "depleted OTS presentation."),
    },
    "load_reduced_but_performance_still_declining": {
        "athlete": ("Even with easier training, performance is still "
                    "dropping - that is why this needs attention now."),
        "coach": ("Performance continues to decline despite load reduction, "
                  "a hallmark of prolonged overtraining rather than acute "
                  "fatigue."),
        "sports_scientist": ("Continued performance decrement under reduced "
                             "load, sustained over weeks, differentiates OTS "
                             "from functional overreaching."),
    },

    # ── travel_jet_lag ────────────────────────────────────────────────
    # (these two tags belong to travel_jet_lag, verified against scenarios.py)
    "low_load_but_poor_hrv_wellness": {
        "athlete": ("Your training is lighter than normal, but travel and "
                    "disrupted sleep make recovery harder right now. This is "
                    "travel-related rather than a sign of overreaching."),
        "coach": ("Suppressed HRV and wellness at reduced load reflect "
                  "circadian disruption from travel rather than training "
                  "overload."),
        "sports_scientist": ("Dissociation between low external load and "
                             "suppressed autonomic markers is attributable to "
                             "circadian misalignment following travel, an "
                             "expected physiological response that resolves "
                             "with resynchronisation."),
    },
    "non_training_cause_of_suppression": {
        "athlete": ("Something other than training is driving these numbers "
                    "down, so easing off training alone won't fix it."),
        "coach": ("The suppression has a non-training cause; adjusting load "
                  "alone will not resolve it."),
        "sports_scientist": ("Suppression is attributable to a non-training "
                             "stressor; load modification alone is unlikely "
                             "to normalise markers."),
    },

    # ── travel_jet_lag ────────────────────────────────────────────────
    # NOTE: verify the exact tags for travel_jet_lag in scenarios.py and add
    # them here before enabling this scenario in the rewrite.

    # ── altitude_camp ─────────────────────────────────────────────────
    # NOTE: altitude_camp currently declares signal_conflicts=[] in
    # scenarios.py despite being conceptually an override scenario (hypoxic
    # suppression at normal load). Declare these two tags on SCENARIO_ALTITUDE
    # to activate this rationale -- see the addendum patch.
    "hypoxic_suppression_not_training_load": {
        "athlete": ("Your recovery numbers are low, but altitude itself is "
                    "the cause rather than your training. This is an expected "
                    "response while you adapt to the thinner air."),
        "coach": ("HRV suppression at altitude reflects hypoxic stress rather "
                  "than training overload. Interpret against altitude-"
                  "adaptation norms, not standard load thresholds."),
        "sports_scientist": ("Autonomic suppression is attributable to "
                             "hypoxic exposure rather than training stress — "
                             "an expected physiological response during "
                             "altitude acclimatisation. Standard overreaching "
                             "criteria do not apply unmodified."),
    },
    "low_load_but_altitude_physiological_cost": {
        "athlete": ("Even at normal training loads, altitude makes every "
                    "session harder on your body."),
        "coach": ("External load metrics understate the true strain at "
                  "altitude - physiological cost is elevated independent of "
                  "workload."),
        "sports_scientist": ("External load underrepresents internal load "
                             "under hypoxic conditions; physiological cost is "
                             "elevated independent of mechanical work."),
    },

    # ── high_acwr_stable_physiology ───────────────────────────────────
    # Inverse direction: load looks alarming, physiology says it is tolerated.
    "acwr_danger_but_hrv_normal": {
        "athlete": ("Your training load is high, but your recovery numbers "
                    "are holding up well - your body is coping with it."),
        "coach": ("ACWR is in the danger zone but HRV remains stable, "
                  "indicating the load is currently well tolerated rather "
                  "than causing physiological strain."),
        "sports_scientist": ("Elevated ACWR without autonomic disturbance "
                             "indicates the load is being tolerated; the "
                             "workload ratio overstates risk in the presence "
                             "of stable physiological markers."),
    },
    "acwr_danger_but_wellness_stable": {
        "athlete": ("Even with heavy training, you're sleeping and feeling "
                    "fine - that is a good sign."),
        "coach": ("Wellness remains stable despite elevated workload, "
                  "supporting continued tolerance of the current programme."),
        "sports_scientist": ("Preserved subjective wellness alongside "
                             "elevated ACWR supports adequate adaptive "
                             "capacity at the current load."),
    },

    # ── double_session_accumulation ───────────────────────────────────
    # Daily metrics look acceptable; the problem is session DENSITY.
    "acwr_moderate_but_recovery_inadequate": {
        "athlete": ("Your overall load looks reasonable, but you aren't "
                    "getting enough recovery between sessions."),
        "coach": ("ACWR is only moderate, but recovery between sessions is "
                  "inadequate - the issue is session density rather than "
                  "total workload."),
        "sports_scientist": ("Moderate ACWR masks insufficient inter-session "
                             "recovery; aggregate load metrics do not capture "
                             "the density-driven recovery deficit."),
    },
    "daily_load_ok_but_session_density_problematic": {
        "athlete": ("Each day's training is fine on its own - it's how "
                    "close together the sessions are that's the problem."),
        "coach": ("Daily load is acceptable; the problem is the frequency of "
                  "double-session days compressing recovery windows."),
        "sports_scientist": ("Daily external load is within range, but "
                             "session density compresses recovery windows "
                             "below the threshold for adequate restoration."),
    },

    # ── wellness_crash_normal_load ────────────────────────────────────
    # ── early_overreaching ────────────────────────────────────────────
    # ACWR is acute/chronic, so CONSISTENTLY high load across all four weeks
    # gives acute ~= chronic and ACWR ~= 1.0. Sustained overload is therefore
    # invisible to the ratio by construction -- the fatigue shows only in HRV
    # and wellness. This is the clinically important case where ACWR fails as
    # a detector, not a sampling error.
    "acwr_normal_but_sustained_accumulation": {
        "athlete": ("Your workload ratio looks normal, but you have been "
                    "training hard for several weeks without a real break. "
                    "The fatigue has built up gradually rather than from any "
                    "single hard week."),
        "coach": ("ACWR appears normal because load has been consistently "
                  "high rather than spiking — acute and chronic loads have "
                  "converged. The accumulated multi-week load is the concern, "
                  "not the ratio."),
        "sports_scientist": ("ACWR is insensitive to sustained overload: with "
                             "consistently elevated load, acute and chronic "
                             "windows converge toward unity. Cumulative "
                             "multi-week exposure, evidenced by progressive "
                             "autonomic and subjective decline, is the "
                             "operative signal here rather than the ratio."),
    },

    "acwr_normal_but_wellness_crashed": {
        "athlete": ("Your training load is normal, so what you're feeling is "
                    "likely coming from outside training."),
        "coach": ("Wellness has deteriorated on a normal training load, "
                  "pointing to a non-training stressor rather than "
                  "overreaching."),
        "sports_scientist": ("Wellness deterioration without corresponding "
                             "load elevation indicates an extrinsic stressor "
                             "rather than training-induced overreaching."),
    },
    "load_acceptable_but_hrv_suppressed": {
        "athlete": ("Training load is fine, but your recovery score says "
                    "your body is under strain from somewhere."),
        "coach": ("Acceptable load with suppressed HRV - investigate "
                  "non-training stressors before modifying the programme."),
        "sports_scientist": ("Autonomic suppression despite acceptable "
                             "external load warrants investigation of "
                             "extrinsic stressors."),
    },

    # ── illness_return ────────────────────────────────────────────────
    # ACWR is acute/chronic. After a layoff the 28-day chronic denominator is
    # depleted, so even modest acute load produces a high ratio. The number
    # overstates risk: it reflects the absence of recent training history
    # rather than dangerous loading.
    "acwr_inflated_by_depleted_chronic_load": {
        "athlete": ("Your workload ratio looks high, but that is because you "
                    "trained very little while ill — the comparison is "
                    "against an unusually low baseline rather than a sign "
                    "you are doing too much."),
        "coach": ("The elevated ACWR reflects a depleted chronic load from "
                  "the illness layoff rather than excessive acute loading. "
                  "Interpret against graduated-return expectations, not "
                  "standard ratio thresholds."),
        "sports_scientist": ("ACWR elevation is denominator-driven: the "
                             "28-day chronic window is artificially "
                             "suppressed by the illness period, inflating the "
                             "ratio independent of acute load. Standard "
                             "thresholds do not apply during graduated "
                             "return."),
    },
    "graduated_return_expected_ratio_elevation": {
        "athlete": ("A rising ratio is expected as you build back up — it "
                    "should settle as your training history rebuilds."),
        "coach": ("Ratio elevation is an expected feature of graduated "
                  "return-to-play and resolves as chronic load rebuilds."),
        "sports_scientist": ("Transient ratio elevation is an expected "
                             "artefact of graduated return; normalisation "
                             "follows chronic-load reconstitution."),
    },
}


def get_rationale(signal_conflict_tags: List[str],
                  audience: str,
                  max_items: int = 2) -> List[str]:
    """Return per-audience rationale sentences for a scenario's declared tags.

    Args:
        signal_conflict_tags: ScenarioConfig.signal_conflicts (already declared
                              in scenarios.py; currently unread).
        audience: 'athlete' | 'coach' | 'sports_scientist'
        max_items: cap sentences so briefs stay in the target length band.
    """
    out = []
    for tag in (signal_conflict_tags or []):
        entry = CONFLICT_RATIONALE.get(tag)
        if entry:
            text = entry.get(audience) or entry.get("coach")
            if text:
                out.append(text)
        if len(out) >= max_items:
            break
    return out


def has_rationale_for(signal_conflict_tags: List[str]) -> bool:
    """True if at least one declared tag has rationale prose defined.

    Use this to gate the rewrite scenario-by-scenario: a scenario whose tags
    are all unmapped will silently produce no rationale, which the consistency
    checker will then flag. Assert on this during regeneration.
    """
    return any(t in CONFLICT_RATIONALE for t in (signal_conflict_tags or []))


def unmapped_tags(all_scenarios: Dict) -> List[str]:
    """List declared tags that have no rationale prose yet.

    Pass ALL_SCENARIOS from scenarios.py. Anything returned here is a scenario
    that will still fail the consistency check after the rewrite.
    """
    missing = []
    for _, cfg in all_scenarios.items():
        for tag in getattr(cfg, "signal_conflicts", []) or []:
            if tag not in CONFLICT_RATIONALE and tag not in missing:
                missing.append(tag)
    return missing


# ── Checker coupling ──────────────────────────────────────────────────
# Add these to consistency_check.OVERRIDE_RATIONALE_PATTERNS. They must match
# the prose above; if you edit a rationale sentence, update its pattern here.
RATIONALE_PATTERNS_FOR_CHECKER = [
    # environmental / heat
    r"expected (environmental|physiological|thermoregulatory) response",
    r"attributable to (thermoregulatory|hypoxic|circadian|a non-training)",
    r"altitude itself is the cause",
    r"expected response while you adapt",
    r"altitude makes every session harder",
    r"heat makes each session cost",
    r"understate the true strain",
    r"underrepresents internal load",
    r"standard (overreaching )?criteria do not apply",
    # generic override
    r"rather than (a sign of )?overreaching",
    r"rather than training[- ](induced|overload|error)",
    r"rather than a training error",
    # youth / growth
    r"developmentally expected",
    r"growth[- ]related",
    r"maturation[- ]adjusted",
    r"growing (quickly|this fast)",
    r"thresholds are (not appropriate|invalid)",
    r"usual training guidelines don't fit",
    # OTS
    r"not a fitness concern",
    r"reflects accumulated chronic fatigue",
    r"does not indicate recovery",
    r"cannot sustain more",
    r"performance is still dropping",
    r"hallmark of prolonged overtraining",
    r"performance decrement under reduced load",
    r"continues to decline despite load reduction",
    # travel / non-training cause
    r"travel[- ]related",
    r"circadian (disruption|misalignment)",
    r"disrupted sleep make recovery harder",
    r"non[- ]training (cause|stressor)",
    r"extrinsic stressor",
    r"won't fix it",
    # tolerated-load (inverse direction)
    r"(well )?tolerated rather than",
    r"is being tolerated",
    r"coping with it",
    r"recovery numbers are holding up",
    r"overstates risk",
    r"supports (continued tolerance|adequate adaptive capacity)",
    r"supporting continued tolerance",
    r"remains stable despite",
    r"masks insufficient",
    r"do not capture",
    r"that is a good sign",
    # session density
    r"session density",
    # early_overreaching / sustained accumulation
    r"without a real break",
    r"built up gradually",
    r"rather than spiking",
    r"accumulated multi-week load",
    r"insensitive to sustained overload",
    r"cumulative\s+multi-week exposure",
    r"between sessions",
    r"recovery windows",
    r"close together the sessions",
    # wellness-crash
    r"coming from outside training",
    r"under strain from somewhere",
    r"against an unusually low baseline",
    r"depleted chronic load",
    r"denominator-driven",
    r"graduated[- ]return",
    r"expected as you build back up",
    r"chronic[- ]load reconstitution",
]


if __name__ == "__main__":
    # Smoke test against the heat scenario's real declared tags.
    heat_tags = ["low_load_intensity_but_high_physiological_cost",
                 "environmental_not_training_cause"]
    for aud in ("athlete", "coach", "sports_scientist"):
        print(f"\n--- heat_acclimatization [{aud}] ---")
        for line in get_rationale(heat_tags, aud):
            print(" ", line)

    # Verify the emitted prose satisfies the checker patterns.
    import re
    print("\n--- checker coupling ---")
    for aud in ("athlete", "coach", "sports_scientist"):
        text = " ".join(get_rationale(heat_tags, aud)).lower()
        hit = next((p for p in RATIONALE_PATTERNS_FOR_CHECKER
                    if re.search(p, text)), None)
        print(f"{aud:<18} rationale detected: {bool(hit)}  ({hit})")