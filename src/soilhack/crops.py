"""soilhack.crops — knowledge-based crop-suitability from published requirement ranges.

HONESTY / SCOPE
---------------
This layer is KNOWLEDGE-BASED, not a trained model. Each crop has parameter ranges (temperature,
annual rainfall, soil pH) compiled from the FAO EcoCrop crop-requirement database and standard
agronomic references. Suitability here is a transparent rule (Liebig's law of the minimum), NOT a
validated ML prediction — we never attach a "model accuracy" to it.

The VALIDATED ML number in this project stays on the REAL crop labels (ESA WorldCereal, 3 crops),
served by TabPFN-3.5 (see predict.py / RESULTS.md).

Ranges (per crop): optimal and absolute limits for temperature (deg C), annual rainfall (mm/yr),
and soil pH. A value inside [opt_min, opt_max] scores 1; outside the absolute limits scores 0;
in between it scores linearly.
"""
from __future__ import annotations

# crop: (group, t_opt_min, t_opt_max, t_abs_min, t_abs_max,
#        r_opt_min, r_opt_max, r_abs_min, r_abs_max, ph_min, ph_max)
# temperature degC | rainfall mm/yr | pH.
REQUIREMENTS = {
    "Maize":         ("Cereal",   18, 32, 10, 45,  500, 1200,  300, 2500, 5.5, 7.5),
    "Rice":          ("Cereal",   20, 35, 15, 40, 1000, 2000,  700, 3000, 5.0, 7.0),
    "Sorghum":       ("Cereal",   25, 35, 15, 45,  400,  800,  250, 1500, 5.5, 8.5),
    "Pearl millet":  ("Cereal",   25, 35, 15, 45,  300,  700,  200, 1200, 5.5, 8.0),
    "Wheat":         ("Cereal",   15, 25,  5, 35,  400,  900,  250, 1600, 6.0, 8.0),
    "Barley":        ("Cereal",   12, 25,  3, 35,  350,  800,  200, 1500, 6.0, 8.5),
    "Teff":          ("Cereal",   15, 27, 10, 35,  450,  800,  300, 1200, 5.5, 7.5),
    "Cassava":       ("Root",     20, 30, 15, 40,  700, 1500,  400, 2500, 4.5, 7.5),
    "Sweet potato":  ("Root",     20, 30, 12, 40,  500, 1200,  300, 2000, 5.0, 7.0),
    "Potato":        ("Root",     15, 22,  5, 30,  500,  800,  300, 1600, 5.0, 6.5),
    "Yam":           ("Root",     25, 30, 20, 35, 1000, 1800,  700, 2500, 5.5, 7.0),
    "Groundnut":     ("Legume",   25, 30, 15, 40,  500, 1200,  300, 2000, 5.5, 7.0),
    "Cowpea":        ("Legume",   25, 35, 15, 45,  350,  900,  250, 2000, 5.5, 7.5),
    "Soybean":       ("Legume",   20, 30, 10, 40,  450,  900,  300, 1800, 6.0, 7.5),
    "Common bean":   ("Legume",   18, 27, 10, 35,  400,  800,  300, 1500, 5.5, 7.5),
    "Chickpea":      ("Legume",   15, 30,  5, 35,  350,  700,  250, 1200, 6.0, 8.5),
    "Cotton":        ("Cash",     25, 35, 15, 45,  500, 1200,  300, 2000, 5.5, 8.5),
    "Sugarcane":     ("Cash",     25, 35, 15, 40, 1000, 1800,  700, 3000, 5.5, 8.0),
    "Coffee":        ("Cash",     15, 24, 10, 30, 1200, 2000,  800, 3000, 5.0, 6.5),
    "Cocoa":         ("Cash",     24, 30, 20, 34, 1200, 2500,  800, 4000, 5.0, 7.0),
    "Banana/plantain": ("Perennial", 24, 30, 18, 35, 1000, 2000,  600, 3000, 5.5, 7.5),
    "Mango":         ("Perennial", 24, 30, 15, 40,  500, 1500,  300, 2500, 5.5, 7.5),
    "Tomato":        ("Vegetable", 18, 27, 10, 35,  400,  900,  250, 2000, 5.5, 7.0),
}

SOURCE = ("Parameter ranges compiled from the FAO EcoCrop crop-requirement database and standard "
          "agronomic references; a knowledge-based filter, not a trained prediction.")


def _trap(x, opt_min, opt_max, abs_min, abs_max):
    """Trapezoidal membership in [0,1]: 1 inside the optimal band, 0 outside the absolute limits."""
    if x is None:
        return None
    if opt_min <= x <= opt_max:
        return 1.0
    if x < abs_min or x > abs_max:
        return 0.0
    if x < opt_min:
        return (x - abs_min) / (opt_min - abs_min)
    return (abs_max - x) / (abs_max - opt_max)


def crop_groups():
    groups = {}
    for c, v in REQUIREMENTS.items():
        groups.setdefault(v[0], []).append(c)
    return groups


def assess(crop, tmean, prec_annual, ph):
    """Return a suitability verdict for one crop from the point's climate + soil.

    Uses Liebig's law of the minimum: the score is the WEAKEST factor, and the limiting factor is
    reported (so the farmer sees WHY). All inputs real (NASA POWER temp/precip, SoilGrids pH).
    """
    if crop not in REQUIREMENTS:
        return None
    _, to0, to1, ta0, ta1, ro0, ro1, ra0, ra1, ph0, ph1 = REQUIREMENTS[crop]
    ts = _trap(tmean, to0, to1, ta0, ta1)
    rs = _trap(prec_annual, ro0, ro1, ra0, ra1)
    ps = _trap(ph, ph0, ph1, ph0 - 1.0, ph1 + 1.0) if ph is not None else None
    factors = {"temperature": ts, "rainfall": rs, "pH": ps}
    present = {k: v for k, v in factors.items() if v is not None}
    if not present:
        return None
    score = min(present.values())
    limiting = min(present, key=present.get)
    if score >= 0.7:
        verdict = "Suitable"
    elif score >= 0.4:
        verdict = "Marginal"
    else:
        verdict = "Not suitable"
    return {
        "crop": crop, "group": REQUIREMENTS[crop][0], "verdict": verdict,
        "score": round(float(score), 3),
        "factors": {k: (round(v, 3) if v is not None else None) for k, v in factors.items()},
        "limiting_factor": limiting,
        "reason": _reason(limiting, tmean, prec_annual, ph, REQUIREMENTS[crop]),
        "source": SOURCE,
    }


def _reason(limiting, tmean, prec, ph, r):
    _, to0, to1, ta0, ta1, ro0, ro1, ra0, ra1, ph0, ph1 = r
    if limiting == "temperature":
        return f"mean temp {tmean:.1f}°C vs optimal {to0}-{to1}°C"
    if limiting == "rainfall":
        return f"annual rainfall {prec:.0f} mm vs optimal {ro0}-{ro1} mm"
    if limiting == "pH":
        return f"soil pH {ph:.1f} vs tolerated {ph0}-{ph1}"
    return ""


def rank(tmean, prec_annual, ph, top=8):
    """Assess every crop and return the ranked list (best first)."""
    out = []
    for c in REQUIREMENTS:
        a = assess(c, tmean, prec_annual, ph)
        if a:
            out.append(a)
    out.sort(key=lambda a: -a["score"])
    return out[:top] if top else out


def alternatives(crop, tmean, prec_annual, ph, n=3):
    """If the queried crop is unsuitable/marginal, suggest the best-suited crops (prefer same group,
    then any)."""
    ranked = rank(tmean, prec_annual, ph, top=None)
    grp = REQUIREMENTS.get(crop, [None])[0]
    same = [a for a in ranked if a["crop"] != crop and a["group"] == grp and a["score"] >= 0.4]
    other = [a for a in ranked if a["crop"] != crop and a["score"] >= 0.4]
    picks = same[:n]
    if len(picks) < n:
        for a in other:
            if a["crop"] not in {p["crop"] for p in picks}:
                picks.append(a)
            if len(picks) >= n:
                break
    return picks[:n]
