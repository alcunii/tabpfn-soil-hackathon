#!/usr/bin/env python3
"""R1 — Real, global soil feature table from the open ISRIC SoilGrids v2.0 REST API.

Proves the "real soil, free, worldwide, no Kaggle" pillar of the project:
query actual soil properties at real lat/lon for US + Africa sites, assemble a table,
and measure coverage + latency. Writes results/real_soilgrids_R1.json + a CSV sample.

RUNS ON THE WORKSTATION (needs internet; stdlib only + pandas).
"""
import json
import os
import ssl
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RES, exist_ok=True)
os.makedirs(RAW, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0 (soilhack)"}

# real sites: US (Kentucky + corn belt) and Africa (Ghana, Kenya, Nigeria)
SITES = [
    ("US_KY_Lexington",   38.05, -84.50),
    ("US_KY_BowlingGreen",36.99, -86.44),
    ("US_IA_cornbelt",    42.03, -93.60),
    ("US_IL_cornbelt",    40.10, -88.20),
    ("US_NE_cornbelt",    41.30, -98.00),
    ("GH_Accra",           5.60,  -0.19),
    ("GH_Kumasi",          6.69,  -1.62),
    ("GH_Tamale",          9.40,  -0.84),
    ("KE_Nairobi",        -1.29,  36.82),
    ("KE_Kisumu",         -0.09,  34.77),
    ("NG_Kano",           12.00,   8.52),
    ("NG_Ibadan",          7.38,   3.90),
    ("TZ_Dodoma",         -6.16,  35.75),
    ("ZA_Bloemfontein",  -29.09,  26.16),
]
PROPS = ["phh2o", "soc", "clay", "sand", "silt", "nitrogen", "cec", "bdod", "socd"]
DEPTH = "0-5cm"


def query_soilgrids(lat, lon):
    q = "&".join([f"property={p}" for p in PROPS])
    url = (f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
           f"lon={lon}&lat={lat}&{q}&depth={DEPTH}&value=mean")
    t0 = time.time()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        obj = json.loads(r.read().decode())
    dt = time.time() - t0
    vals = {}
    for layer in obj["properties"]["layers"]:
        name = layer["name"]
        d0 = layer["depths"][0]["values"].get("mean")
        factor = layer["unit_measure"].get("d_factor", 1)
        vals[name] = None if d0 is None else d0 / factor if factor else d0
        vals[name + "_raw"] = d0
        vals[name + "_factor"] = factor
    return vals, dt


def main():
    rows = []
    fail = 0
    t_total = time.time()
    for name, lat, lon in SITES:
        try:
            vals, dt = query_soilgrids(lat, lon)
            row = {"site": name, "lat": lat, "lon": lon, "latency_s": round(dt, 2)}
            row.update({p: vals.get(p) for p in PROPS})
            rows.append(row)
            print(f"  {name:<18} {dt:4.1f}s  pH={vals.get('phh2o')} SOC={vals.get('soc')} "
                  f"clay={vals.get('clay')} N={vals.get('nitrogen')}", flush=True)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  {name:<18} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(RAW, "real_soilgrids_R1.csv")
    df.to_csv(csv, index=False)

    coverage = {p: int(df[p].notna().sum()) if p in df else 0 for p in PROPS}
    out = {
        "experiment": "R1_real_soilgrids_api",
        "source": "ISRIC SoilGrids v2.0 REST API (rest.isric.org), depth 0-5cm, mean",
        "n_sites": len(SITES),
        "n_ok": len(rows),
        "n_fail": fail,
        "properties": PROPS,
        "coverage_by_property": coverage,
        "latency_s_median": round(float(df["latency_s"].median()), 2) if len(df) else None,
        "wall_s": round(time.time() - t_total, 1),
        "csv": csv,
        "note": "Real global soil properties at real coordinates, free, no API key.",
    }
    with open(os.path.join(RES, "real_soilgrids_R1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if len(rows) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
