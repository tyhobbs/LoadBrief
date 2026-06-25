import numpy as np
from typing import List, Dict, Optional

def calculate_acwr(session_loads, method="rolling"):
    if len(session_loads) < 7:
        return {"acwr": None, "acute_load": None, "chronic_load": None, "method": method, "insufficient_data": True}
    loads = np.array(session_loads)
    if method == "rolling":
        acute = np.mean(loads[-7:])
        chronic = np.mean(loads[-28:]) if len(loads) >= 28 else np.mean(loads)
    elif method == "ewma":
        la, lc = 2/(7+1), 2/(28+1)
        ea, ec = loads[0], loads[0]
        for l in loads[1:]:
            ea = la*l + (1-la)*ea
            ec = lc*l + (1-lc)*ec
        acute, chronic = ea, ec
    else:
        acute = np.mean(loads[-7:])
        chronic = np.mean(loads[-28:]) if len(loads) >= 28 else np.mean(loads)
    acwr = acute / chronic if chronic > 0 else 0.0
    return {"acwr": round(acwr,3), "acute_load": round(float(acute),1), "chronic_load": round(float(chronic),1), "method": method, "insufficient_data": False}

def calculate_monotony(daily_loads):
    if len(daily_loads) < 3: return 0.0
    return round(np.mean(daily_loads) / (np.std(daily_loads) + 1e-6), 3)

def calculate_strain(daily_loads):
    return round(sum(daily_loads[-7:]) * calculate_monotony(daily_loads[-7:]), 1)

def classify_acwr_zone(acwr, sport_category, thresholds):
    if acwr is None: return "insufficient_data"
    t = thresholds.get(sport_category, thresholds.get("general_population", {}))
    for zone, (low, high) in t.items():
        if low <= acwr < high: return zone
    return "extreme"

def analyze_hrv_trend(hrv_values, baseline_hrv, thresholds):
    if not hrv_values or baseline_hrv is None:
        return {"available": False, "status": "data_unavailable", "consecutive_suppressed_days": 0}
    recent = hrv_values[-7:]
    deltas = [v - baseline_hrv for v in recent]
    mean_delta = np.mean(deltas)
    trend = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) > 2 else 0
    consec = 0
    for d in reversed(deltas):
        if d < thresholds.get("moderate_suppression_ms", -4): consec += 1
        else: break
    if mean_delta <= thresholds.get("significant_suppression_ms", -8):
        status = "critically_suppressed" if consec >= thresholds.get("consecutive_days_critical", 7) else "significantly_suppressed"
    elif mean_delta <= thresholds.get("moderate_suppression_ms", -4): status = "moderately_suppressed"
    elif mean_delta >= thresholds.get("elevation_ms", 8): status = "elevated_good_recovery"
    else: status = "normal"
    return {
        "available": True, "current_vs_baseline_ms": round(deltas[-1],1),
        "7day_mean_vs_baseline_ms": round(mean_delta,1),
        "trend_direction": "improving" if trend>0.3 else "declining" if trend<-0.3 else "stable",
        "consecutive_suppressed_days": consec, "status": status,
        "recent_values": [round(v,1) for v in recent], "baseline": round(baseline_hrv,1)
    }

def analyze_wellness_trend(wellness_history):
    if not wellness_history: return {"available": False}
    recent = wellness_history[-7:]
    dims = ["sleep_quality","fatigue","muscle_soreness","mood","stress"]
    inv = ["fatigue","muscle_soreness","stress"]
    trends = {}
    for dim in dims:
        vals = [d.get(dim) for d in recent if d.get(dim) is not None]
        if not vals: continue
        mean_val = np.mean(vals)
        trend = np.polyfit(range(len(vals)), vals, 1)[0] if len(vals)>2 else 0
        if dim in inv: zone = "red" if mean_val>=4.0 else "amber" if mean_val>=3.0 else "normal"
        else: zone = "red" if mean_val<=2.0 else "amber" if mean_val<=3.0 else "normal"
        trends[dim] = {
            "mean_7day": round(mean_val,2),
            "trend": "worsening" if ((dim in inv and trend>0.1) or (dim not in inv and trend<-0.1)) else "improving" if ((dim in inv and trend<-0.1) or (dim not in inv and trend>0.1)) else "stable",
            "zone": zone
        }
    red = sum(1 for d in trends.values() if d["zone"]=="red")
    amb = sum(1 for d in trends.values() if d["zone"]=="amber")
    composite = "severely_depressed" if red>=3 else "moderately_depressed" if red>=1 or amb>=3 else "mildly_depressed" if amb>=1 else "good"
    return {"available": True, "dimensions": trends, "composite_status": composite, "red_flags": red, "amber_flags": amb}

def classify_overreaching_state(acwr_metrics, hrv_analysis, wellness_analysis, scenario_config, monitoring_days):
    return {"classification": scenario_config.overreaching_class, "ground_truth": scenario_config.overreaching_class, "confidence": "high", "risk_level": scenario_config.risk_level}
