"""soilhack.predict — the recommendation engine (real data -> TabPFN-3.5 -> ranked crops).

Flow for a query (lat, lon):
  1. fetch real soil   -> ISRIC SoilGrids v2.0 REST (6 properties, 0-5 cm)
  2. fetch real climate -> NASA POWER daily point API (7 summary covariates)
  3. assemble a CONTEXT from the real reference library (the k nearest observed points)
  4. fit TabPFN-3.5 on that context (NO training), predict crop probabilities
  5. return ranked crops + calibrated confidence + the soil/climate evidence

This module has NO web dependency; the app imports it. It fetches live data only for the query
point (one SoilGrids call + one POWER call); the reference library ships as a CSV.
"""
import json
import os
import ssl
import statistics
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SOIL = ["phh2o", "soc", "clay", "nitrogen", "cec", "bdod"]
CLIM = ["tmean", "trange", "prec_annual", "prec_cv", "gdd5", "frost_days", "aridity"]
FEATURES = SOIL + CLIM

_CTX = ssl.create_default_context()  # verify TLS (the public APIs have valid certs)
_UA = {"User-Agent": "Mozilla/5.0 (soilhack)"}

REF_CANDIDATES = [
    os.environ.get("SOILHACK_LIBRARY", ""),
    os.path.join(ROOT, "data", "samples", "reference_library.csv"),
    os.path.join(HERE, "reference_library.csv"),
]

_LIB = None
_MODEL = None


# ----------------------------------------------------------------- data fetch
def fetch_soil(lat, lon, tries=2):
    q = "&".join(f"property={p}" for p in SOIL)
    url = (f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
           f"lon={lon:.4f}&lat={lat:.4f}&{q}&depth=0-5cm&value=mean")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=15, context=_CTX) as r:
                o = json.loads(r.read().decode())
            out = {}
            for layer in o["properties"]["layers"]:
                v = layer["depths"][0]["values"].get("mean")
                f = layer["unit_measure"].get("d_factor", 1) or 1
                out[layer["name"]] = None if v is None else v / f
            if any(out.get(p) is not None for p in SOIL):
                return out
        except Exception:
            time.sleep(0.5)
    return {p: None for p in SOIL}


def fetch_climate(lat, lon, start="20190101", end="20231231", tries=2):
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?"
           f"parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR&community=AG&"
           f"longitude={lon:.4f}&latitude={lat:.4f}&start={start}&end={end}&format=JSON")
    p = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
                p = json.loads(r.read().decode())["properties"]["parameter"]
            break
        except Exception:
            time.sleep(0.5)
    if p is None:
        return {c: None for c in CLIM}

    def vals(key):
        return [v for v in p.get(key, {}).values() if isinstance(v, (int, float)) and v > -900]
    t, tmax, tmin, pr = vals("T2M"), vals("T2M_MAX"), vals("T2M_MIN"), vals("PRECTOTCORR")
    if not t or not pr:
        return {c: None for c in CLIM}
    n_yr = 5.0
    tmean = sum(t) / len(t)
    trange = (sum(tmax) / len(tmax)) - (sum(tmin) / len(tmin)) if (tmax and tmin) else 0.0
    months = {}
    for i, v in enumerate(p["PRECTOTCORR"].values()):
        if isinstance(v, (int, float)) and v > -900:
            months.setdefault(i // 30, 0.0)
            months[i // 30] += v
    mv = list(months.values())
    return {
        "tmean": round(tmean, 2), "trange": round(trange, 2),
        "prec_annual": round(sum(pr) / n_yr, 1),
        "prec_cv": round(statistics.pstdev(mv) / statistics.mean(mv), 3) if mv and statistics.mean(mv) else 0.0,
        "gdd5": round(sum(max(x - 5.0, 0.0) for x in t) / n_yr, 0),
        "frost_days": round(sum(1 for x in tmin if x < 0.0) / n_yr, 1),
        "aridity": round((sum(pr) / n_yr) / (tmean + 10.0), 2) if (tmean + 10.0) > 0 else 0.0,
    }


# ----------------------------------------------------------------- model
def _load_model(fit_cache=True):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import os as _os
    _os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")
    from tabpfn import TabPFNClassifier
    device = "cpu"
    ckpt = _os.environ.get("SOILHACK_CKPT")
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    kw = {}
    if fit_cache:
        # resident model: preprocess the context once, keep the cache on device, so each later
        # single-row predict is fast (the documented pattern for serving one row at a time).
        kw = {"fit_mode": "fit_with_cache", "keep_cache_on_device": True}
    if ckpt and _os.path.exists(ckpt):
        _MODEL = TabPFNClassifier(device=device, model_path=ckpt, **kw)
    else:
        _MODEL = TabPFNClassifier.create_default_for_version("v3.5", device=device, **kw)
    return _MODEL


def load_library():
    global _LIB
    if _LIB is not None:
        return _LIB
    for p in REF_CANDIDATES:
        if os.path.exists(p):
            _LIB = pd.read_csv(p)
            return _LIB
    raise FileNotFoundError("reference_library.csv not found in " + "; ".join(REF_CANDIDATES))


def _classes():
    return sorted(load_library()["crop"].unique())


_MODEL_WARMED = False


def warm(context_rows=None):
    """Fit the resident model ONCE on the reference library (call at app startup).

    After warm(), recommend() only runs a single predict per query (fast)."""
    global _MODEL_WARMED
    lib = load_library()
    ctx = lib if context_rows is None else lib.sample(min(context_rows, len(lib)), random_state=42)
    X = ctx[FEATURES].to_numpy("float32")
    codes = {c: i for i, c in enumerate(_classes())}
    y = np.array([codes[c] for c in ctx["crop"]])
    model = _load_model(fit_cache=True)
    model.fit(X, y)
    _MODEL_WARMED = True
    return {"context_rows": int(len(ctx)), "classes": _classes()}


def nearest_context(lat, lon, k=300, region=None, ensure_classes=True):
    """Nearest-k observed rows; guarantee every crop class is present (a 1-class context cannot
    rank crops). k is in CONTEXT ROWS (TabPFN cost scales with context, not query count)."""
    lib = load_library()
    d = lib[lib["region"] == region] if (region and region in ("US", "Africa")) else lib
    d = d.copy()
    d["_dist"] = (d["lat"] - lat) ** 2 + (d["lon"] - lon) ** 2
    ctx = d.nsmallest(k, "_dist")
    if ensure_classes:
        have = set(ctx["crop"].unique())
        missing = [c for c in lib["crop"].unique() if c not in have]
        if missing:
            extra = d[d["crop"].isin(missing)].nsmallest(10 * len(missing), "_dist")
            ctx = pd.concat([ctx, extra]).drop_duplicates()
    return ctx


def recommend(lat, lon, k=300, region=None, soil=None, climate=None, explain=True):
    """Return ranked crop recommendations with confidence + evidence."""
    t0 = time.time()
    soil = soil or fetch_soil(lat, lon)
    climate = climate or fetch_climate(lat, lon)
    query = {**{p: soil.get(p) for p in SOIL}, **{c: climate.get(c) for c in CLIM}}
    missing = [f for f in FEATURES if query.get(f) is None]

    # fallback: fill any missing feature from the NEAREST real library field (real observed values),
    # so a demo never breaks when a live API is slow. The fallback is reported, never hidden.
    filled_from_library = []
    if missing:
        nn = nearest_context(lat, lon, k=1)
        for f in missing:
            try:
                v = float(nn.iloc[0][f])
            except Exception:
                continue
            if v == v:  # not NaN
                query[f] = v
                filled_from_library.append(f)

    ctx = nearest_context(lat, lon, k=k, region=region)
    Xq = np.array([[np.nan if query[f] is None else query[f] for f in FEATURES]], dtype="float32")
    classes = _classes()
    if _MODEL_WARMED:
        # resident model path: predict only (fast; context fitted once at warm())
        proba = _MODEL.predict_proba(Xq)[0]
    else:
        codes = {c: i for i, c in enumerate(classes)}
        ytr = np.array([codes[c] for c in ctx["crop"]])
        model = _load_model(fit_cache=False)
        model.fit(ctx[FEATURES].to_numpy("float32"), ytr)
        proba = model.predict_proba(Xq)[0]
    order = np.argsort(-proba)
    ranked = [{"crop": classes[i], "confidence": round(float(proba[i]), 3)} for i in order]

    # context region mix (used for the honesty note)
    ctx_regions = ctx["region"].value_counts().to_dict()
    out = {
        "lat": round(lat, 4), "lon": round(lon, 4),
        "recommendations": ranked,
        "confidence_gap": round(float(proba[order[0]] - proba[order[1]]), 3) if len(order) > 1 else None,
        "context": {"k": int(len(ctx)), "region_mix": ctx_regions,
                    "mean_dist_deg": round(float(np.sqrt(ctx["_dist"]).mean()), 3),
                    "crop_mix": ctx["crop"].value_counts().to_dict()},
        "evidence": {
            "soil": {p: round(soil[p], 2) if soil.get(p) is not None else None for p in SOIL},
            "climate": {c: climate.get(c) for c in CLIM},
        },
        "effective": {f: {"value": (round(query[f], 3) if query.get(f) is not None else None),
                          "source": ("nearest-field" if f in filled_from_library else "live")}
                      for f in FEATURES},
        "missing_features": [f for f in FEATURES if query.get(f) is None],
        "filled_from_library": filled_from_library,
        "data_note": ("live APIs for all 13 features" if not filled_from_library
                      else f"live APIs + nearest-field fallback for {len(filled_from_library)} feature(s)"),
        "model": "TabPFN-3.5 (zero-shot; no training)",
        "elapsed_s": round(time.time() - t0, 2),
    }
    if explain:
        out["explain"] = explain_query(ctx, query)
    return out


def explain_query(ctx, query, top=3):
    """Cheap, honest explanation: which features put the query at an edge of the LIBRARY's range.

    Uses stable library-wide stats (not the tiny context) to compute a z-score of the query,
    and reports whether the query is inside the observed range. NOT a causal claim.
    """
    lib = load_library()
    notes = []
    zs = {}
    for f in FEATURES:
        col = lib[f].to_numpy(float)
        mu, sd = np.nanmean(col), np.nanstd(col)
        qv = query.get(f)
        if qv is None or sd == 0:
            continue
        zs[f] = round(float((qv - mu) / sd), 2)
    for f, z in sorted(zs.items(), key=lambda t: -abs(t[1]))[:top]:
        if abs(z) >= 1.5:
            direction = "high" if z > 0 else "low"
            notes.append(f"{f} is unusually {direction} vs the observed library (z={z:+.1f})")
    if not notes:
        notes.append("all soil/climate values are within the observed library range")
    return notes


# ----------------------------------------------------------------- competence
def competence(lat, lon):
    """Honest competence note from the measured experiments (see RESULTS.md).

    Zero-shot cross-continent transfer was ≈ the majority baseline; a few local points fix it.
    We report whether the query's nearest context points are from the SAME continent as before,
    and state the measured caveat.
    """
    ctx = nearest_context(lat, lon, k=25)
    mix = ctx["region"].value_counts().to_dict()
    dom = max(mix, key=mix.get)
    return {
        "nearest_context_regions": mix,
        "note": ("Trained on REAL observations. Measured: zero-shot transfer across continents is "
                 "weak (≈majority baseline); with ~5 local observations the 3-class recommendation "
                 "is essentially solved, carried mostly by climate. Treat a prediction far from the "
                 "context as a prior, not a fact."),
        "dominant_region": dom,
    }


# ----------------------------------------------------------------- suitability (knowledge + real)
def suitability(lat, lon, crop=None, soil=None, climate=None):
    """Assess whether a crop suits a point.

    Two clearly-labelled layers:
      * KNOWLEDGE layer (all crops): FAO EcoCrop ranges + Liebig minimum -> suitability + reason.
        This is a transparent rule, NOT a trained prediction (no accuracy attached).
      * VALIDATED-ML cross-check (only the 3 real-label crops): TabPFN-3.5's prediction on real
        labels, so the user sees when the learned model agrees with the knowledge rule.
    """
    from soilhack import crops as C
    soil = soil or fetch_soil(lat, lon)
    climate = climate or fetch_climate(lat, lon)
    tmean = climate.get("tmean")
    prec = climate.get("prec_annual")
    ph = soil.get("phh2o")

    def _fill(v, key):
        return v

    if crop:
        a = C.assess(crop, tmean, prec, ph)
        if a is None:
            return {"error": f"unknown crop '{crop}'", "known": list(C.REQUIREMENTS)}
        out = {"lat": lat, "lon": lon, "assessment": a,
               "alternatives": C.alternatives(crop, tmean, prec, ph, n=3),
               "layer": "knowledge-based (FAO EcoCrop ranges) - NOT a trained prediction",
               "evidence": {"tmean": tmean, "prec_annual": prec, "ph": ph}}
        # cross-check against the validated ML layer ONLY if this crop is one of the 3 real crops
        if crop in _classes():
            r = recommend(lat, lon, soil=soil, climate=climate)
            match = next((x for x in r["recommendations"] if x["crop"] == crop), None)
            out["ml_crosscheck"] = {"crop": crop, "tabpfn_confidence": match["confidence"] if match else None,
                                    "note": "TabPFN-3.5 on REAL labels (this crop IS in the validated set)"}
        else:
            out["ml_crosscheck"] = {"note": "no validated labels for this crop in the open data; "
                                            "suitability is the knowledge rule only"}
        return out

    return {"lat": lat, "lon": lon, "ranked": C.rank(tmean, prec, ph, top=12),
            "groups": C.crop_groups(),
            "layer": "knowledge-based (FAO EcoCrop ranges) - NOT a trained prediction",
            "evidence": {"tmean": tmean, "prec_annual": prec, "ph": ph}}


if __name__ == "__main__":
    import sys
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 9.40
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else -0.84
    print(json.dumps(recommend(lat, lon), indent=1))
