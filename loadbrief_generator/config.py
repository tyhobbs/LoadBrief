ACWR_THRESHOLDS = {
    "team_field_sports": {"undertraining":(0.0,0.8),"sweet_spot":(0.8,1.3),"caution":(1.3,1.5),"danger":(1.5,2.0),"extreme":(2.0,99.0)},
    "team_court_sports": {"undertraining":(0.0,0.8),"sweet_spot":(0.8,1.3),"caution":(1.3,1.5),"danger":(1.5,2.0),"extreme":(2.0,99.0)},
    "endurance_sports": {"undertraining":(0.0,0.8),"sweet_spot":(0.8,1.4),"caution":(1.4,1.6),"danger":(1.6,2.0),"extreme":(2.0,99.0)},
    "sprint_power_sports": {"undertraining":(0.0,0.8),"sweet_spot":(0.8,1.3),"caution":(1.3,1.5),"danger":(1.5,2.0),"extreme":(2.0,99.0)},
    "combat_sports": {"undertraining":(0.0,0.7),"sweet_spot":(0.7,1.2),"caution":(1.2,1.5),"danger":(1.5,2.0),"extreme":(2.0,99.0)},
    "racket_sports": {"undertraining":(0.0,0.8),"sweet_spot":(0.8,1.3),"caution":(1.3,1.5),"danger":(1.5,2.0),"extreme":(2.0,99.0)},
    "technical_sports": {"undertraining":(0.0,0.8),"sweet_spot":(0.8,1.3),"caution":(1.3,1.5),"danger":(1.5,2.0),"extreme":(2.0,99.0)},
    "youth_athletes": {"undertraining":(0.0,0.8),"sweet_spot":(0.8,1.2),"caution":(1.2,1.4),"danger":(1.4,1.8),"extreme":(1.8,99.0)},
    "general_population": {"undertraining":(0.0,0.8),"sweet_spot":(0.8,1.3),"caution":(1.3,1.5),"danger":(1.5,2.0),"extreme":(2.0,99.0)},
    "military_tactical": {"undertraining":(0.0,0.8),"sweet_spot":(0.8,1.3),"caution":(1.3,1.5),"danger":(1.5,2.0),"extreme":(2.0,99.0)},
}
HRV_THRESHOLDS = {
    "significant_suppression_ms": -8, "moderate_suppression_ms": -4,
    "normal_range_ms": 4, "elevation_ms": 8,
    "consecutive_days_concern": 3, "consecutive_days_critical": 7
}
WELLNESS_NORMS = {
    "sleep_quality":   {"normal":(3.0,4.5),"amber":(2.0,3.0),"red":(1.0,2.0)},
    "fatigue":         {"normal":(1.5,3.0),"amber":(3.0,4.0),"red":(4.0,5.0)},
    "muscle_soreness": {"normal":(1.5,3.0),"amber":(3.0,4.0),"red":(4.0,5.0)},
    "mood":            {"normal":(3.0,4.5),"amber":(2.0,3.0),"red":(1.0,2.0)},
    "stress":          {"normal":(1.5,3.0),"amber":(3.0,4.0),"red":(4.0,5.0)}
}
OVERREACHING_CRITERIA = {
    "functional_overreaching": {"acwr_min":1.3,"duration_days_max":14,"recovery_days":(3,7)},
    "non_functional_overreaching": {"acwr_min":1.3,"duration_days_min":14,"recovery_weeks":(2,6)},
    "overtraining_syndrome": {"duration_weeks_min":6,"recovery_months":(2,12)}
}
MONOTONY_THRESHOLDS = {"acceptable":(0.0,2.0),"concerning":(2.0,3.0),"high_risk":(3.0,99.0)}
SPORT_CATEGORIES = {
    "team_field_sports": ["soccer","rugby_union","rugby_league","american_football","australian_rules_football","field_hockey","lacrosse","handball","netball","gaelic_football","hurling"],
    "team_court_sports": ["basketball","volleyball","ice_hockey","water_polo"],
    "endurance_sports": ["marathon_running","middle_distance_running","cycling_road","triathlon","rowing","swimming_distance","cross_country_skiing","cycling_track_endurance","canoe_kayak_sprint"],
    "sprint_power_sports": ["sprinting_100m","sprinting_200m","sprinting_400m","olympic_weightlifting","powerlifting","throws_athletics","jumps_athletics","swimming_sprint","cycling_track_sprint"],
    "combat_sports": ["wrestling","judo","boxing","mma","taekwondo","fencing"],
    "racket_sports": ["tennis","badminton","squash","table_tennis","padel"],
    "technical_sports": ["gymnastics_artistic","gymnastics_rhythmic","diving","figure_skating","alpine_skiing"],
    "youth_athletes": ["youth_soccer","youth_basketball","youth_swimming","youth_athletics","youth_rugby","youth_multi_sport"],
    "general_population": ["recreational_running","recreational_cycling","crossfit","general_fitness","obstacle_course_racing","recreational_swimming"],
    "military_tactical": ["army_soldier","navy_special_forces","police_officer","firefighter","pararescue"]
}
TRAINING_PHASES = ["off_season","pre_season_early","pre_season_late","in_season_early","in_season_mid","in_season_late","taper","competition","post_season_recovery","return_from_injury","return_from_illness"]
DATA_LEVELS = {
    1: {"name":"minimal","description":"Basic wearable","available_signals":["session_rpe","training_load","sleep_duration"],"missing_signals":["hrv","wellness_questionnaire","gps_load"]},
    2: {"name":"moderate","description":"Collegiate","available_signals":["session_rpe","training_load","sleep_duration","hrv","wellness_questionnaire"],"missing_signals":["gps_load"]},
    3: {"name":"comprehensive","description":"Professional","available_signals":["session_rpe","training_load","sleep_duration","hrv","wellness_questionnaire","gps_load","accelerometry","heart_rate_zones"],"missing_signals":[]},
    4: {"name":"research_grade","description":"Research","available_signals":["session_rpe","training_load","sleep_duration","hrv","wellness_questionnaire","gps_load","accelerometry","heart_rate_zones","lactate_threshold","vo2max","force_plate","body_composition"],"missing_signals":[]}
}
AUDIENCE_PROFILES = {
    "athlete": {"flesch_kincaid_target":(6,9),"technical_density_max":0.05,"word_count_range":(150,250),"tone":"supportive_direct","avoid_terms":["ACWR","HRV","AU","monotony index"],"preferred_terms":{"ACWR":"training spike ratio","HRV":"recovery score","AU":"training units","non_functional_overreaching":"overtraining warning"}},
    "coach": {"flesch_kincaid_target":(9,12),"technical_density_max":0.12,"word_count_range":(250,400),"tone":"clinical_actionable","avoid_terms":[],"preferred_terms":{}},
    "sports_scientist": {"flesch_kincaid_target":(12,16),"technical_density_max":0.25,"word_count_range":(350,600),"tone":"technical_precise","avoid_terms":[],"preferred_terms":{}},
    "api": {"format":"structured_json","word_count_range":(0,0),"tone":"machine_readable"}
}
