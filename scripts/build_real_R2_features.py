#!/usr/bin/env python3
"""Build the real soil feature table (R2) — INCREMENTAL and resumable.

Samples ISRIC SoilGrids v2.0 for each real WorldCereal label point, appending each result to
data/raw/real_R2_features.csv as it lands and skipping points already done, so a re-run resumes.
Writes a progress log the caller can poll.

    python3 scripts/build_real_R2_features.py
"""
import csv
import json
import os
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)
LOG = os.path.join(ROOT, "logs", "R2_features.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
OUT = os.path.join(RAW, "real_R2_features.csv")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0 (soilhack)"}
PROPS = ["phh2o", "soc", "clay", "nitrogen", "cec", "bdod"]
FIELDS = ["crop", "region", "country", "state", "lat", "lon"] + PROPS + ["soil_ok"]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def query_sg(lat, lon, tries=3):
    q = "&".join(f"property={p}" for p in PROPS)
    url = (f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
           f"lon={lon}&lat={lat}&{q}&depth=0-5cm&value=mean")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25, context=ctx) as r:
                o = json.loads(r.read().decode())
            out = {}
            for layer in o["properties"]["layers"]:
                v = layer["depths"][0]["values"].get("mean")
                f = layer["unit_measure"].get("d_factor", 1) or 1
                out[layer["name"]] = None if v is None else v / f
            return out
        except Exception:
            time.sleep(0.4 + k)
    return None


def keyof(r):
    return f"{r['crop']}|{r['region']}|{r['lat']}|{r['lon']}"


def main():
    log("start (incremental)")
    df = pd.read_csv(os.path.join(ROOT, "data", "samples", "worldcereal_real_sample.csv"))
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(keyof(r) for _, r in prev.iterrows())
        log(f"resuming: {len(done)} already sampled")
    todo = [i for i, r in df.iterrows() if keyof(r) not in done]
    log(f"total {len(df)}, todo {len(todo)}")

    newfile = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if newfile:
        w.writeheader()
    n = len(done)

    def work(i):
        r = df.iloc[i]
        s = query_sg(r["lat"], r["lon"])
        rec = {"crop": r["crop"], "region": r["region"], "country": r["country"],
               "state": r["state"], "lat": r["lat"], "lon": r["lon"], "soil_ok": 0}
        if s and any(s.get(p) is not None for p in PROPS):
            rec.update({p: s.get(p) for p in PROPS})
            rec["soil_ok"] = 1 if all(s.get(p) is not None for p in PROPS) else 0
        return rec

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            w.writerow(rec)
            fh.flush()
            n += 1
            if n % 100 == 0:
                log(f"{n}/{len(df)} sampled")
    fh.close()

    feat = pd.read_csv(OUT)
    has = feat[feat["soil_ok"] == 1]
    summary = {
        "experiment": "R2_feature_build",
        "n_total": int(len(feat)), "n_soil_ok": int(len(has)),
        "coverage": round(len(has) / len(feat), 3),
        "by_region": has["region"].value_counts().to_dict(),
        "by_crop": has["crop"].value_counts().to_dict(),
        "by_region_crop": has.groupby(["region", "crop"]).size().to_dict(),
        "csv": OUT,
    }
    # JSON-safe keys
    summary["by_region_crop"] = {f"{k[0]}/{k[1]}": int(v) for k, v in summary["by_region_crop"].items()}
    with open(os.path.join(RES, "real_R2_features_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
