# narrative/templates.py
# All template string libraries for monitoring narrative generation.
# Multiple variants per condition prevent repetitive text
# across 50,000 training examples.

# ── Header Templates ──────────────────────────────────────────────────
HEADER_TEMPLATES = [
    "Athlete monitoring summary — {period}:",
    "Load management report for {sport} athlete ({period}):",
    "Training monitoring data — {period} review:",
    "Weekly load management brief ({period}):",
    "Monitoring snapshot: {sport}, {period}:",
    "Performance monitoring report — {period}:",
    "{sport} athlete monitoring data ({period}):",
    "Training load review — {period}:",
    "Athlete readiness assessment ({period}):",
    "Monitoring status report: {period}:"
]

# ── Athlete Context Templates ─────────────────────────────────────────
ATHLETE_CONTEXT_TEMPLATES = [
    "{sport} {position}, {level} level, {training_age} years training experience. "
    "Currently in {phase}.",

    "Athlete profile: {level} {sport} {position}, "
    "training age {training_age} years, phase: {phase}.",

    "{level} {sport} player ({position}). "
    "Training background: {training_age} years. "
    "Current phase: {phase}.",

    "Subject: {sport} {position} ({level}), "
    "{training_age}-year training background, "
    "{phase} phase.",

    "{phase} monitoring for {level} {sport} {position} "
    "({training_age} years structured training)."
]

# ── ACWR Description Templates ────────────────────────────────────────
ACWR_TEMPLATES = {
    "undertraining": [
        "The ACWR of {acwr} sits below the optimal training range, "
        "suggesting load may be insufficient to drive adaptation.",

        "Training load metrics indicate potential undertraining — "
        "ACWR of {acwr} is below the recommended 0.8 minimum.",

        "Acute:chronic ratio of {acwr} indicates the athlete may "
        "be undertrained. Chronic adaptation may be at risk.",

        "The ACWR ({acwr}) is below the sweet spot, potentially "
        "indicating detraining risk."
    ],
    "sweet_spot": [
        "The ACWR of {acwr} sits comfortably within the optimal "
        "training range (0.8–1.3), indicating well-managed load.",

        "Training load is well-periodized — ACWR of {acwr} reflects "
        "a healthy balance between fitness and fatigue.",

        "The acute:chronic ratio ({acwr}) is in the sweet spot, "
        "indicating appropriate progressive loading.",

        "Load metrics are healthy: ACWR of {acwr} within the "
        "recommended range.",

        "ACWR of {acwr} — training load is building appropriately "
        "without excessive fatigue accumulation."
    ],
    "caution": [
        "The ACWR of {acwr} is approaching the upper caution "
        "threshold — intensified monitoring is recommended.",

        "Training load is elevated (ACWR {acwr}), approaching "
        "the caution zone. Close monitoring advised.",

        "The acute:chronic ratio of {acwr} sits in the caution zone. "
        "Load is increasing faster than the optimal rate.",

        "ACWR of {acwr} warrants attention — the athlete is in the "
        "caution range and should be monitored closely.",

        "Load metrics show ACWR of {acwr} — entering the caution "
        "zone where injury risk begins to elevate."
    ],
    "danger": [
        "The ACWR of {acwr} has entered the danger zone (above 1.5), "
        "significantly elevating injury risk.",

        "Training load spike has pushed ACWR to {acwr} — "
        "well above established safe thresholds.",

        "An ACWR of {acwr} indicates acute load has substantially "
        "exceeded chronic capacity. Injury risk is elevated.",

        "ACWR of {acwr} is concerning — the athlete's recent load "
        "spike demands immediate management attention.",

        "The acute:chronic ratio ({acwr}) is in the danger zone. "
        "Research suggests materially elevated injury risk at this level."
    ],
    "extreme": [
        "ACWR of {acwr} is at an extreme level — injury risk is "
        "very high. Immediate load reduction is required.",

        "The training load spike has produced a dangerous ACWR "
        "of {acwr}. This level carries substantial injury risk.",

        "At {acwr}, the ACWR has reached an extreme value "
        "requiring urgent management intervention.",

        "Critical: ACWR of {acwr} is well above safe thresholds. "
        "Training must be modified immediately."
    ],
    "insufficient_data": [
        "Insufficient training data for reliable ACWR calculation "
        "(fewer than 7 days of load records available).",

        "ACWR cannot be calculated — less than one week of "
        "consistent training data recorded.",

        "Training load history is incomplete — ACWR calculation "
        "requires at least 7 days of session data."
    ]
}

# ── Raw Load Value Templates ──────────────────────────────────────────
LOAD_VALUE_TEMPLATES = [
    "Acute load (past 7 days): {acute:.0f} AU. "
    "Chronic load (28-day average): {chronic:.0f} AU.",

    "7-day acute load: {acute:.0f} AU | "
    "28-day chronic load: {chronic:.0f} AU.",

    "Recent training load: {acute:.0f} AU acute, "
    "{chronic:.0f} AU chronic.",

    "Load figures: acute {acute:.0f} AU, "
    "chronic {chronic:.0f} AU."
]

# ── Monotony Templates ────────────────────────────────────────────────
MONOTONY_TEMPLATES = {
    "acceptable": [
        "Training monotony is acceptable ({monotony:.2f}) — "
        "session variation is adequate.",

        "Session-to-session variation is sufficient "
        "(monotony index: {monotony:.2f}).",

        "Training monotony ({monotony:.2f}) is within normal range."
    ],
    "concerning": [
        "Training monotony is elevated ({monotony:.2f}) — "
        "sessions may be too similar in load.",

        "Monotony index of {monotony:.2f} suggests insufficient "
        "session variation. Training variety should increase.",

        "The monotony index ({monotony:.2f}) is concerning — "
        "daily loads are too similar, limiting recovery stimulus."
    ],
    "high_risk": [
        "High training monotony detected ({monotony:.2f}) — "
        "sessions are nearly identical, driving fatigue accumulation "
        "independent of total load.",

        "Monotony index of {monotony:.2f} is in the high-risk zone. "
        "Daily load variation must increase immediately.",

        "Critical monotony ({monotony:.2f}): training variety is "
        "dangerously low. Even at normal ACWR, high monotony "
        "significantly elevates injury and illness risk."
    ]
}

# ── HRV Description Templates ─────────────────────────────────────────
HRV_TEMPLATES = {
    "normal": [
        "Morning HRV is stable ({current}ms), within normal "
        "variation of the {baseline}ms individual baseline.",

        "HRV readings are normal — {current}ms is close to "
        "the {baseline}ms baseline (+/- {delta}ms).",

        "Heart rate variability shows no concerning trends, "
        "sitting at {current}ms against a {baseline}ms baseline.",

        "HRV is tracking near baseline ({current}ms vs "
        "{baseline}ms) — autonomic readiness is good.",

        "Morning HRV: {current}ms. Within acceptable range "
        "of individual baseline ({baseline}ms)."
    ],
    "elevated_good_recovery": [
        "Morning HRV is elevated ({current}ms, +{delta}ms above "
        "baseline), indicating excellent autonomic recovery.",

        "HRV is tracking above baseline ({current}ms vs {baseline}ms) "
        "— a positive readiness and recovery signal.",

        "The elevated HRV reading ({current}ms, +{delta}ms above "
        "{baseline}ms baseline) suggests the athlete is well-recovered.",

        "Strong HRV response: {current}ms is {delta}ms above "
        "individual baseline, indicating readiness for load."
    ],
    "moderately_suppressed": [
        "Morning HRV has dipped to {current}ms, {delta}ms below "
        "individual baseline ({baseline}ms).",

        "HRV shows mild suppression: {current}ms versus the "
        "{baseline}ms baseline, a {delta}ms reduction.",

        "A {delta}ms drop in HRV below baseline suggests "
        "early fatigue accumulation.",

        "HRV is moderately suppressed ({current}ms vs {baseline}ms "
        "baseline) — mild autonomic fatigue indicated."
    ],
    "significantly_suppressed": [
        "Morning HRV is significantly suppressed at {current}ms — "
        "{delta}ms below the {baseline}ms individual baseline.",

        "HRV suppression of {delta}ms below baseline ({current}ms "
        "vs {baseline}ms) indicates meaningful physiological fatigue.",

        "A {delta}ms suppression in HRV, sustained for "
        "{days} consecutive days, is a clear fatigue signal.",

        "HRV has been consistently below baseline: {current}ms "
        "against {baseline}ms, a {delta}ms deficit for {days} days.",

        "Significant HRV suppression: {current}ms ({delta}ms below "
        "{baseline}ms baseline, {days} consecutive days)."
    ],
    "critically_suppressed": [
        "HRV has been critically suppressed for {days} consecutive "
        "days — {current}ms against a {baseline}ms baseline "
        "({delta}ms below baseline).",

        "Severe HRV suppression: {delta}ms below baseline for "
        "{days} days ({current}ms vs {baseline}ms). This indicates "
        "significant physiological stress.",

        "The {days}-day run of HRV suppression ({current}ms vs "
        "{baseline}ms) is consistent with non-functional overreaching.",

        "Critical: HRV at {current}ms — {delta}ms below individual "
        "baseline for {days} consecutive mornings."
    ],
    "data_unavailable": [
        "HRV data is not available for this monitoring period.",
        "No HRV measurements were recorded during this block.",
        "HRV tracking was not active for this athlete.",
        "HRV monitoring not available — assessment based on "
        "available signals only."
    ],
    "not_available": [
        "HRV monitoring not available for this athlete.",
        "No HRV device in use — load management based on "
        "RPE and wellness data only."
    ]
}

# ── Wellness Description Templates ───────────────────────────────────
WELLNESS_ALL_NORMAL = [
    "All wellness dimensions are within normal ranges. "
    "Subjective readiness is good.",

    "Wellness questionnaire responses are all normal — "
    "the athlete is reporting good subjective state.",

    "No wellness flags. Sleep, fatigue, soreness, mood, "
    "and stress are all within acceptable ranges.",

    "Wellness is good across all dimensions — no subjective "
    "readiness concerns."
]

WELLNESS_ONE_FLAG = [
    "Wellness is generally good with one flagged dimension: "
    "{dim} is {zone} and trending {trend}.",

    "Overall wellness is normal, though {dim} is flagging "
    "as {zone}. Other dimensions remain acceptable.",

    "One wellness concern: {dim} ({zone} zone, "
    "{trend} trend). All other dimensions are normal."
]

WELLNESS_MULTIPLE_FLAGS = [
    "Multiple wellness dimensions are flagged: {dims}. "
    "Composite wellness status: {composite}.",

    "Wellness is compromised across {n} dimensions: {dims}. "
    "Overall status: {composite}.",

    "The athlete is reporting below-normal wellness in "
    "the following areas: {dims}. "
    "Composite status: {composite}.",

    "Wellness data shows {red_count} red-flag and "
    "{amber_count} amber-flag dimensions: {dims}."
]

WELLNESS_PARTIAL_DATA = [
    "Full wellness questionnaire not available — "
    "sleep data from wearable only.",

    "Wellness monitoring is limited to sleep tracking "
    "for this athlete.",

    "Only sleep duration data available — detailed wellness "
    "assessment requires full questionnaire."
]

# ── RPE Log Templates ─────────────────────────────────────────────────
RPE_LOG_TEMPLATES = [
    "Session RPE log (past 7 days): {rpe_log}",
    "Recent session RPE scores: {rpe_log}",
    "Training load perception (past 7 days, RPE/10): {rpe_log}",
    "Daily RPE readings: {rpe_log}"
]

# ── Sleep Templates ───────────────────────────────────────────────────
SLEEP_TEMPLATES = {
    "good": [
        "Sleep duration averaging {hours:.1f} hours per night — adequate.",
        "Sleep: {hours:.1f} hours average per night (normal range).",
        "Average nightly sleep of {hours:.1f} hours — within healthy range."
    ],
    "borderline": [
        "Sleep duration slightly reduced at {hours:.1f} hours per night.",
        "Average sleep of {hours:.1f} hours per night — slightly below optimal.",
        "Sleep averaging {hours:.1f} hours — marginally below recommended 7-9 hours."
    ],
    "poor": [
        "Sleep is significantly reduced — averaging only {hours:.1f} hours per night.",
        "Poor sleep duration: {hours:.1f} hours average, well below the 7-9 hour target.",
        "Sleep deprivation indicated: {hours:.1f} hours average nightly sleep."
    ]
}

# ── Injury History Note Templates ────────────────────────────────────
INJURY_NOTE_TEMPLATES = [
    "Note: athlete has history of {sites} — "
    "monitor these areas specifically.",

    "Prior injury history: {sites}. "
    "Load management should account for these sites.",

    "Injury flags: previous {sites}. "
    "Exercise selection and progression should be modified accordingly."
]

# ── Special Circumstance Note Templates ──────────────────────────────
ALTITUDE_NOTE_TEMPLATES = [
    "Currently training at altitude ({altitude}m) — "
    "physiological signals modified by hypoxic environment.",

    "Altitude camp context ({altitude}m): "
    "HRV and wellness suppression partially explained by "
    "altitude acclimatization response.",

    "Note: {altitude}m altitude training. "
    "Standard load thresholds may not apply directly."
]

HEAT_NOTE_TEMPLATES = [
    "Training environment: {temp}°C, {humidity}% humidity — "
    "cardiovascular strain elevated beyond what RPE alone suggests.",

    "Heat stress context: {temp}°C/{humidity}% RH. "
    "Physiological cost of sessions is higher than load metrics indicate.",

    "Environmental stressor: extreme heat ({temp}°C). "
    "HRV and wellness suppression partially attributable to "
    "thermoregulatory demand."
]

TRAVEL_NOTE_TEMPLATES = [
    "Recent travel: {zones} time zones crossed. "
    "Circadian disruption may be affecting HRV and wellness.",

    "Jet lag context: {zones}-zone travel {direction}. "
    "Sleep disruption is expected to persist for several days.",

    "International travel note: {zones} time zones. "
    "Non-training factors are contributing to current physiological state."
]

YOUTH_NOTE_TEMPLATES = [
    "Note: youth athlete currently at or near Peak Height Velocity. "
    "Standard load thresholds apply more conservatively.",

    "PHV consideration: athlete is in a growth phase. "
    "Injury risk is elevated relative to standard ACWR thresholds.",

    "Growth stage alert: athlete is near PHV. "
    "Load management requires conservative application of adult norms."
]
