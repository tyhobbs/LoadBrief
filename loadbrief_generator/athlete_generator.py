# athlete_generator.py
# Generates realistic synthetic athlete profiles
# All parameter distributions grounded in published sport science norms

import random
import numpy as np
from typing import Dict, Optional


# ── Sport-specific HRV baselines (ms) ─────────────────────────────────
# Source: Plews et al. 2013, Buchheit 2014, Kiviniemi et al. 2010
# (mean, std) by sport category and level
HRV_BASELINES = {
    "team_field_sports": {
        "recreational":  (55, 10),
        "amateur":       (62, 10),
        "semi_professional": (68, 10),
        "professional":  (75, 12),
        "elite":         (82, 12)
    },
    "team_court_sports": {
        "recreational":  (53, 10),
        "amateur":       (60, 10),
        "semi_professional": (66, 10),
        "professional":  (73, 12),
        "elite":         (79, 12)
    },
    "endurance_sports": {
        "recreational":  (58, 12),
        "amateur":       (68, 12),
        "semi_professional": (78, 12),
        "professional":  (88, 14),
        "elite":         (98, 14)
    },
    "sprint_power_sports": {
        "recreational":  (52, 10),
        "amateur":       (58, 10),
        "semi_professional": (64, 10),
        "professional":  (70, 12),
        "elite":         (76, 12)
    },
    "combat_sports": {
        "recreational":  (54, 10),
        "amateur":       (60, 10),
        "semi_professional": (66, 10),
        "professional":  (72, 12),
        "elite":         (78, 12)
    },
    "racket_sports": {
        "recreational":  (55, 10),
        "amateur":       (61, 10),
        "semi_professional": (67, 10),
        "professional":  (73, 12),
        "elite":         (79, 12)
    },
    "technical_sports": {
        "recreational":  (53, 10),
        "amateur":       (59, 10),
        "semi_professional": (65, 10),
        "professional":  (71, 12),
        "elite":         (77, 12)
    },
    "youth_athletes": {
        "recreational":  (60, 12),
        "amateur":       (65, 12),
        "semi_professional": (70, 12),
        "professional":  (75, 12),
        "elite":         (80, 12)
    },
    "general_population": {
        "recreational":  (48, 12),
        "amateur":       (54, 12),
        "semi_professional": (60, 12),
        "professional":  (66, 12),
        "elite":         (72, 12)
    },
    "military_tactical": {
        "recreational":  (55, 10),
        "amateur":       (60, 10),
        "semi_professional": (65, 10),
        "professional":  (70, 12),
        "elite":         (75, 12)
    }
}

# ── Chronic load baselines (AU) by sport and level ───────────────────
# Source: Gabbett 2016, Hulin et al. 2016
CHRONIC_LOAD_BASELINES = {
    "team_field_sports": {
        "recreational":  (600, 150),
        "amateur":       (900, 200),
        "semi_professional": (1200, 250),
        "professional":  (1600, 300),
        "elite":         (2000, 350)
    },
    "endurance_sports": {
        "recreational":  (700, 150),
        "amateur":       (1100, 200),
        "semi_professional": (1600, 250),
        "professional":  (2200, 350),
        "elite":         (2800, 400)
    },
    "sprint_power_sports": {
        "recreational":  (500, 120),
        "amateur":       (750, 150),
        "semi_professional": (1000, 200),
        "professional":  (1400, 250),
        "elite":         (1800, 300)
    },
    "combat_sports": {
        "recreational":  (600, 150),
        "amateur":       (900, 200),
        "semi_professional": (1300, 250),
        "professional":  (1700, 300),
        "elite":         (2100, 350)
    },
    "racket_sports": {
        "recreational":  (500, 120),
        "amateur":       (800, 150),
        "semi_professional": (1100, 200),
        "professional":  (1500, 250),
        "elite":         (1900, 300)
    },
    "general_population": {
        "recreational":  (300, 100),
        "amateur":       (500, 120),
        "semi_professional": (700, 150),
        "professional":  (900, 200),
        "elite":         (1200, 250)
    },
    "military_tactical": {
        "recreational":  (700, 150),
        "amateur":       (1000, 200),
        "semi_professional": (1400, 250),
        "professional":  (1800, 300),
        "elite":         (2200, 350)
    }
}

# ── Position lookup per sport ─────────────────────────────────────────
POSITIONS = {
    "soccer": ["goalkeeper", "center_back", "fullback",
               "central_midfielder", "attacking_midfielder",
               "winger", "striker"],
    "rugby_union": ["loosehead_prop", "hooker", "tighthead_prop",
                    "lock", "blindside_flanker", "openside_flanker",
                    "number_8", "scrum_half", "fly_half",
                    "inside_centre", "outside_centre",
                    "wing", "fullback"],
    "rugby_league": ["prop", "hooker", "second_row", "loose_forward",
                     "halfback", "stand_off", "centre", "wing", "fullback"],
    "american_football": ["quarterback", "running_back", "wide_receiver",
                          "tight_end", "offensive_lineman",
                          "defensive_lineman", "linebacker",
                          "cornerback", "safety", "kicker"],
    "australian_rules_football": ["forward", "midfielder",
                                   "defender", "ruck"],
    "field_hockey": ["goalkeeper", "defender", "midfielder",
                     "forward"],
    "lacrosse": ["attack", "midfield", "defense", "goalkeeper"],
    "handball": ["goalkeeper", "center_back", "left_back",
                 "right_back", "left_wing", "right_wing", "pivot"],
    "netball": ["goal_shooter", "goal_attack", "wing_attack",
                "centre", "wing_defence", "goal_defence",
                "goal_keeper"],
    "basketball": ["point_guard", "shooting_guard", "small_forward",
                   "power_forward", "center"],
    "volleyball": ["setter", "outside_hitter", "middle_blocker",
                   "opposite", "libero"],
    "ice_hockey": ["center", "left_wing", "right_wing",
                   "defenseman", "goalkeeper"],
    "water_polo": ["goalkeeper", "driver", "hole_set", "wing"],
    "marathon_running": ["open_category"],
    "middle_distance_running": ["800m", "1500m", "steeplechase"],
    "cycling_road": ["climber", "sprinter", "rouleur",
                     "time_trialist", "domestique"],
    "triathlon": ["sprint_distance", "olympic_distance",
                  "half_iron", "full_iron"],
    "rowing": ["single_scull", "double_scull", "quad_scull",
               "coxed_pair", "coxless_pair", "coxed_four",
               "coxless_four", "eight"],
    "swimming_distance": ["freestyle", "backstroke",
                          "breaststroke", "butterfly", "IM"],
    "swimming_sprint": ["freestyle", "backstroke",
                        "breaststroke", "butterfly"],
    "olympic_weightlifting": ["snatch_specialist", "clean_jerk",
                               "all_round"],
    "powerlifting": ["squat_specialist", "bench_specialist",
                     "deadlift_specialist", "all_round"],
    "sprinting_100m": ["open_category"],
    "sprinting_200m": ["open_category"],
    "sprinting_400m": ["open_category"],
    "throws_athletics": ["shot_put", "discus", "hammer", "javelin"],
    "jumps_athletics": ["long_jump", "triple_jump",
                        "high_jump", "pole_vault"],
    "wrestling": ["freestyle", "greco_roman"],
    "judo": ["open_category"],
    "boxing": ["amateur", "professional"],
    "mma": ["open_category"],
    "tennis": ["singles", "doubles"],
    "badminton": ["singles", "doubles", "mixed_doubles"],
    "squash": ["open_category"],
    "gymnastics_artistic": ["floor", "vault", "bars", "beam",
                             "rings", "pommel", "all_round"],
    "alpine_skiing": ["slalom", "giant_slalom",
                      "super_g", "downhill"],
    "cycling_track_sprint": ["sprint", "keirin", "team_sprint"],
    "cycling_track_endurance": ["individual_pursuit",
                                 "team_pursuit", "points_race",
                                 "omnium", "madison"],
    "cross_country_skiing": ["classic", "skate", "biathlon"],
    "canoe_kayak_sprint": ["kayak_single", "kayak_double",
                            "canoe_single", "canoe_double"],
    "youth_soccer": ["goalkeeper", "defender",
                     "midfielder", "forward"],
    "youth_basketball": ["guard", "forward", "center"],
    "youth_swimming": ["freestyle", "backstroke",
                       "breaststroke", "butterfly", "IM"],
    "youth_athletics": ["sprints", "distance",
                        "jumps", "throws", "multi_event"],
    "youth_rugby": ["forward", "back"],
    "youth_multi_sport": ["open_category"],
    "recreational_running": ["open_category"],
    "recreational_cycling": ["open_category"],
    "crossfit": ["open_category"],
    "general_fitness": ["open_category"],
    "obstacle_course_racing": ["open_category"],
    "recreational_swimming": ["open_category"],
    "army_soldier": ["infantry", "special_forces",
                     "support", "officer"],
    "navy_special_forces": ["operator", "support"],
    "police_officer": ["patrol", "tactical", "detective"],
    "firefighter": ["structural", "wildland", "hazmat"],
    "pararescue": ["open_category"],
    "gaelic_football": ["goalkeeper", "defender",
                        "midfielder", "forward"],
    "hurling": ["goalkeeper", "defender",
                "midfielder", "forward"],
    "futsal": ["goalkeeper", "pivot", "winger", "fixo"],
    "beach_volleyball": ["open_category"],
    "padel": ["open_category"],
    "table_tennis": ["singles", "doubles"],
    "fencing": ["foil", "epee", "sabre"],
    "cycling_mtb": ["cross_country", "downhill", "enduro"],
    "taekwondo": ["open_category"],
    "figure_skating": ["singles", "pairs", "ice_dance"],
    "diving": ["platform", "springboard", "synchronised"],
    "gymnastics_rhythmic": ["open_category"],
    "biathlon": ["sprint", "individual", "mass_start", "relay"],
}


class AthleteProfileGenerator:
    """
    Generates realistic synthetic athlete profiles.
    All numerical parameters sampled from published sport science norms.
    """

    def __init__(self, sport_categories: dict,
                 training_phases: list,
                 hrv_thresholds: dict):
        self.sport_categories = sport_categories
        self.training_phases = training_phases
        self.hrv_thresholds = hrv_thresholds

        # Flatten sport list for easy sampling
        self.all_sports = []
        self.sport_to_category = {}
        for category, sports in sport_categories.items():
            for sport in sports:
                self.all_sports.append(sport)
                self.sport_to_category[sport] = category

        self.levels = [
            "recreational", "amateur", "semi_professional",
            "professional", "elite"
        ]
        self.level_weights = [25, 30, 25, 15, 5]  # % distribution

    def generate(self,
                 sport: Optional[str] = None,
                 level: Optional[str] = None,
                 phase: Optional[str] = None) -> Dict:
        """Generate a single athlete profile"""

        # Pick sport
        sport = sport or random.choice(self.all_sports)
        sport_category = self.sport_to_category.get(
            sport, "general_population"
        )

        # Pick level
        level = level or random.choices(
            self.levels, weights=self.level_weights
        )[0]

        # Pick phase
        phase = phase or random.choice(self.training_phases)

        # Pick position
        position = self._sample_position(sport)

        # Pick demographics
        age = self._sample_age(sport_category, level)
        sex = random.choices(
            ["male", "female"],
            weights=[55, 45]
        )[0]

        # Training age (years of structured training)
        training_age = self._sample_training_age(level)

        # Baseline HRV from published norms
        baseline_hrv = self._sample_baseline_hrv(
            sport_category, level, sex
        )

        # Baseline wellness (athlete's normal state)
        baseline_wellness = self._sample_baseline_wellness(level)

        # Baseline chronic load
        chronic_load = self._sample_chronic_load(
            sport_category, level, phase
        )

        # Individual response variability
        # Some athletes are HRV-sensitive, others less so
        hrv_reactivity = max(0.5, np.random.normal(1.0, 0.15))
        rpe_sensitivity = max(0.5, np.random.normal(1.0, 0.12))

        # Injury history flags
        injury_history = self._sample_injury_history(
            sport, age, training_age
        )

        return {
            "sport": sport,
            "sport_category": sport_category,
            "position": position,
            "level": level,
            "age": age,
            "sex": sex,
            "training_age_years": training_age,
            "phase": phase,
            "baseline_hrv": round(baseline_hrv, 1),
            "baseline_wellness": baseline_wellness,
            "baseline_chronic_load": round(chronic_load, 0),
            "hrv_reactivity": round(hrv_reactivity, 2),
            "rpe_sensitivity": round(rpe_sensitivity, 2),
            "injury_history": injury_history,
            "athlete_id": self._generate_id()
        }

    def _sample_position(self, sport: str) -> str:
        positions = POSITIONS.get(sport, ["open_category"])
        return random.choice(positions)

    def _sample_age(self, sport_category: str,
                    level: str) -> int:
        """Age distributions vary by sport category and level"""
        age_ranges = {
            "youth_athletes": (12, 17),
            "endurance_sports": (18, 40),
            "team_field_sports": (17, 35),
            "sprint_power_sports": (17, 33),
            "combat_sports": (18, 35),
            "general_population": (18, 65),
            "military_tactical": (18, 45)
        }
        low, high = age_ranges.get(sport_category, (18, 35))

        # Elite athletes tend to be in prime years
        if level in ["professional", "elite"]:
            low = max(low, 18)
            high = min(high, 32)

        # Ensure valid range — low can exceed high after clamping
        if low > high:
            low = high
        return random.randint(low, high)

    def _sample_training_age(self, level: str) -> float:
        """Years of structured training"""
        ranges = {
            "recreational": (0.5, 2),
            "amateur": (1, 5),
            "semi_professional": (3, 10),
            "professional": (7, 18),
            "elite": (10, 22)
        }
        low, high = ranges.get(level, (1, 5))
        return round(random.uniform(low, high), 1)

    def _sample_baseline_hrv(self, sport_category: str,
                              level: str, sex: str) -> float:
        """
        Sample baseline HRV from sport/level norms.
        Females typically have slightly higher HRV than males
        (Nunan et al. 2010).
        """
        category_baselines = HRV_BASELINES.get(
            sport_category,
            HRV_BASELINES["general_population"]
        )
        mean, std = category_baselines.get(
            level,
            category_baselines["recreational"]
        )

        # Female HRV adjustment (+5ms average, Nunan et al. 2010)
        if sex == "female":
            mean += 5

        hrv = np.random.normal(mean, std)
        return max(20, min(150, hrv))  # physiological bounds

    def _sample_baseline_wellness(self, level: str) -> Dict:
        """
        Sample baseline (normal) wellness scores.
        Higher level athletes tend to have more stable wellness.
        """
        # Base scores — 3.5 is "good but not perfect" normal
        stability = {
            "recreational": 0.4,
            "amateur": 0.35,
            "semi_professional": 0.3,
            "professional": 0.25,
            "elite": 0.2
        }.get(level, 0.35)

        return {
            "sleep_quality": round(
                np.clip(np.random.normal(3.8, stability), 2.5, 5.0), 1
            ),
            "fatigue": round(
                np.clip(np.random.normal(2.0, stability), 1.0, 3.5), 1
            ),
            "muscle_soreness": round(
                np.clip(np.random.normal(2.0, stability), 1.0, 3.5), 1
            ),
            "mood": round(
                np.clip(np.random.normal(3.8, stability), 2.5, 5.0), 1
            ),
            "stress": round(
                np.clip(np.random.normal(2.0, stability), 1.0, 3.5), 1
            )
        }

    def _sample_chronic_load(self, sport_category: str,
                              level: str,
                              phase: str) -> float:
        """
        Sample baseline chronic training load (AU).
        Phase affects the expected chronic load level.
        """
        category = CHRONIC_LOAD_BASELINES.get(
            sport_category,
            CHRONIC_LOAD_BASELINES["general_population"]
        )
        mean, std = category.get(level, category["recreational"])

        # Phase modifiers
        phase_modifiers = {
            "off_season": 0.6,
            "pre_season_early": 0.8,
            "pre_season_late": 1.1,
            "in_season_early": 1.0,
            "in_season_mid": 1.0,
            "in_season_late": 0.95,
            "taper": 0.7,
            "competition": 0.75,
            "post_season_recovery": 0.5,
            "return_from_injury": 0.4,
            "return_from_illness": 0.3
        }
        modifier = phase_modifiers.get(phase, 1.0)

        load = np.random.normal(mean * modifier, std * 0.3)
        return max(100, load)

    def _sample_injury_history(self, sport: str,
                               age: int,
                               training_age: float) -> Dict:
        """
        Generate injury history flags.
        Older athletes with more training years have higher injury history.
        """
        injury_probability = min(
            0.8,
            0.1 + (age - 18) * 0.02 + training_age * 0.03
        )

        history = {
            "has_prior_injury": random.random() < injury_probability,
            "sites": [],
            "currently_managing": False
        }

        if history["has_prior_injury"]:
            # Sport-specific common injury sites
            sport_injuries = {
                "soccer": ["ankle", "knee_acl", "hamstring",
                           "groin", "calf"],
                "rugby_union": ["shoulder", "knee", "hamstring",
                                "neck", "ankle"],
                "american_football": ["knee_acl", "shoulder",
                                      "ankle", "concussion"],
                "running": ["knee_it_band", "shin_splints",
                            "achilles", "plantar_fascia",
                            "stress_fracture"],
                "swimming": ["shoulder_rotator_cuff", "knee",
                             "lower_back"],
                "default": ["knee", "ankle", "shoulder",
                            "lower_back", "hamstring"]
            }
            sites = sport_injuries.get(
                sport,
                sport_injuries["default"]
            )
            n_sites = random.choices([1, 2, 3],
                                     weights=[60, 30, 10])[0]
            history["sites"] = random.sample(
                sites, min(n_sites, len(sites))
            )
            history["currently_managing"] = (
                random.random() < 0.2
            )

        return history

    def _generate_id(self) -> str:
        """Generate anonymous athlete ID"""
        return f"ATH_{random.randint(100000, 999999)}"

    def generate_batch(self, n: int,
                       sport: Optional[str] = None) -> list:
        """Generate multiple athlete profiles"""
        return [self.generate(sport=sport) for _ in range(n)]
