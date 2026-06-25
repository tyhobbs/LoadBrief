# simulator/data_levels.py
# Filters full monitoring metrics down to what each
# data completeness level athlete would actually have.
# Level 1 = basic wearable (RPE + sleep only)
# Level 2 = collegiate (RPE + HRV + wellness)
# Level 3 = professional (all of the above + GPS)
# Level 4 = research grade (everything)

import copy
from typing import Dict


class DataLevelFilter:
    """
    Applies data completeness filtering to monitoring metrics.
    Simulates the reality that different athletes have access
    to different monitoring tools.
    """

    def __init__(self, data_levels: dict):
        self.data_levels = data_levels

    def apply(self,
              acwr_metrics: Dict,
              hrv_analysis: Dict,
              wellness_analysis: Dict,
              data_level: int) -> Dict:
        """
        Returns filtered metrics dict based on data level.
        Missing signals are replaced with unavailable markers.
        """
        level_config = self.data_levels.get(data_level,
                                             self.data_levels[2])
        available = level_config["available_signals"]

        filtered = {
            "acwr": self._filter_acwr(acwr_metrics, available),
            "hrv": self._filter_hrv(hrv_analysis, available),
            "wellness": self._filter_wellness(
                wellness_analysis, available
            ),
            "gps": self._filter_gps(available),
            "data_level": data_level,
            "data_level_name": level_config["name"],
            "available_signals": available
        }

        return filtered

    def _filter_acwr(self, acwr_metrics: Dict,
                     available: list) -> Dict:
        """
        ACWR is always available if RPE is being tracked.
        RPE is available at all levels.
        """
        if "session_rpe" not in available:
            return {
                "available": False,
                "reason": "no_rpe_tracking"
            }

        result = copy.deepcopy(acwr_metrics)
        result["available"] = True
        return result

    def _filter_hrv(self, hrv_analysis: Dict,
                    available: list) -> Dict:
        """HRV only available at Level 2+"""
        if "hrv" not in available:
            return {
                "available": False,
                "reason": "no_hrv_device",
                "status": "data_unavailable",
                "note": ("HRV monitoring not available. "
                         "Interpretation based on load and "
                         "wellness signals only.")
            }

        result = copy.deepcopy(hrv_analysis)
        result["available"] = True
        return result

    def _filter_wellness(self, wellness_analysis: Dict,
                         available: list) -> Dict:
        """
        Full wellness questionnaire available at Level 2+.
        Level 1 gets sleep duration only (from wearable).
        """
        if "wellness_questionnaire" not in available:
            # Level 1 — sleep only from wearable
            return {
                "available": True,
                "partial": True,
                "note": ("Full wellness questionnaire not available. "
                         "Sleep data only from wearable device."),
                "composite_status": "partial_data",
                "dimensions": {
                    "sleep_quality": wellness_analysis.get(
                        "dimensions", {}
                    ).get("sleep_quality", {"mean_7day": None,
                                            "zone": "unknown"})
                },
                "red_flags": 0,
                "amber_flags": 0
            }

        result = copy.deepcopy(wellness_analysis)
        result["available"] = True
        result["partial"] = False
        return result

    def _filter_gps(self, available: list) -> Dict:
        """GPS/external load only available at Level 3+"""
        if "gps_load" not in available:
            return {
                "available": False,
                "reason": "no_gps_device",
                "note": ("External load data (GPS) not available. "
                         "Load monitoring based on session RPE method.")
            }

        return {
            "available": True,
            "note": "GPS external load data available"
        }
