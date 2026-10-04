#!/usr/bin/env python3
"""Build CLIMATE covariates for the real crop-label points.

Source: NASA POWER daily point API (free, no key, no batching). Dedupes label points to a
0.25-degree grid (climate is smooth at this scale), fetches daily 2019-2023 per unique cell,
expands back to one row per sample point.

Features: mean temp, temp range, annual precip, precip seasonality (monthly CV),
growing-degree-days (base 5C), frost days, aridity index.

    python3 scripts/build_climate_features.py
"""
import json
import os
import ssl
import statistics
import sys
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
RES = os.path.join(ROOT, "results")
os.makedirs(RAW, exist_ok=True)
LOG = os.path.join(ROOT, "logs", "climate_features.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
OUT = os.path.join(RAW, "real_climate_features.csv")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0 (soilhack)"}
START, END = "20190101", "20231231"
GRID = 0.25
FILL = -900.0
FIELDS = ["lat", "lon", "cell", "tmean", "trange", "prec_annual", "prec_cv",
          "gdd5", "frost_days", "aridity"]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def cell_of(lat, lon):
    return f"{round(lat / GRID):d}|{round(lon / GRID):d}"


def power_point(lat, lon, tries=4):
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?"
           f"parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR&community=AG&"
           f"longitude={lon:.4f}&latitude={lat:.4f}&start={START}&end={END}&format=JSON")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(2 + 2 * k)
    return None


def summarize(p):
    def vals(key):
        return [v for v in p.get(key, {}).values() if isinstance(v, (int, float)) and v > FILL]
    t, tmax, tmin, pr = vals("T2M"), vals("T2M_MAX"), vals("T2M_MIN"), vals("PRECTOTCORR")
    if not t or not pr:
        return None
    n_yr = 5.0
    tmean = sum(t) / len(t)
    trange = (sum(tmax) / len(tmax)) - (sum(tmin) / len(tmin)) if (tmax and tmin) else 0.0
    prec_annual = sum(pr) / n_yr
    months = {}
    for i, v in enumerate(p["PRECTOTCORR"].values()):
        if isinstance(v, (int, float)) and v > FILL:
            months.setdefault(i // 30, 0.0)
            months[i // 30] += v
    mv = list(months.values())
    prec_cv = (statistics.pstdev(mv) / statistics.mean(mv)) if mv and statistics.mean(mv) else 0.0
    gdd5 = sum(max(x - 5.0, 0.0) for x in t) / n_yr
    frost = sum(1 for x in tmin if x < 0.0) / n_yr
    aridity = prec_annual / (tmean + 10.0) if (tmean + 10.0) > 0 else 0.0
    return {"tmean": round(tmean, 2), "trange": round(trange, 2), "prec_annual": round(prec_annual, 1),
            "prec_cv": round(prec_cv, 3), "gdd5": round(gdd5, 0), "frost_days": round(frost, 1),
            "aridity": round(aridity, 2)}


def main():
    log("start climate build (NASA POWER, grid-deduped)")
    pts = pd.read_csv(os.path.join(ROOT, "data", "samples", "worldcereal_real_sample.csv"))
    pts["cell"] = [cell_of(a, b) for a, b in zip(pts["lat"], pts["lon"])]
    cells = pts.drop_duplicates("cell")[["cell", "lat", "lon"]].reset_index(drop=True)
    log(f"{len(pts)} points -> {len(cells)} unique {GRID}deg cells")

    cell_clim = {}
    for i, row in cells.iterrows():
        o = power_point(row["lat"], row["lon"])
        if o and "properties" in o:
            s = summarize(o["properties"]["parameter"])
            if s:
                cell_clim[row["cell"]] = s
        if (i + 1) % 25 == 0:
            log(f"cells {i+1}/{len(cells)} (got {len(cell_clim)})")
        time.sleep(0.4)

    rows = []
    for _, r in pts.iterrows():
        s = cell_clim.get(r["cell"])
        if not s:
            continue
        rec = {"lat": r["lat"], "lon": r["lon"], "cell": r["cell"]}
        rec.update(s)
        rows.append(rec)
    feat = pd.DataFrame(rows)
    feat.to_csv(OUT, index=False)
    summary = {
        "experiment": "climate_feature_build",
        "source": "NASA POWER daily point API 2019-2023 (community AG), 0.25deg-deduped",
        "n_points": int(len(pts)), "n_cells": int(len(cells)),
        "n_cells_ok": int(len(cell_clim)), "n_rows_out": int(len(feat)),
        "features": [f for f in FIELDS if f not in ("lat", "lon", "cell")],
        "csv": OUT,
    }
    with open(os.path.join(RES, "real_climate_features_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
