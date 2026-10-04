#!/usr/bin/env python3
"""R2 — REAL soil -> REAL crop label, with a geographic transfer test.

Inputs (both real, both open, no Kaggle):
  * crop labels  : ESA WorldCereal point sample (data/samples/worldcereal_real_sample.csv)
  * soil features: ISRIC SoilGrids v2.0 REST API (real soil at each real lat/lon)

Question: can a TabPFN-3.5 model recommend the grown crop from SOIL ALONE, and does it
TRANSFER between the US and Africa? The scientific answer (measured, not assumed) is what
the project reports. Also computes the majority-class baseline for every split.

RUNS ON THE WORKSTATION. Writes results/real_R2_*.json.
"""
import json
import os
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from soilhack import metrics as sm  # noqa: E402
from soilhack import tabpfn_runner as tr  # noqa: E402

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0 (soilhack)"}
PROPS = [("phh2o", "0-5cm"), ("soc", "0-5cm"), ("clay", "0-5cm"),
         ("nitrogen", "0-5cm"), ("cec", "0-5cm"), ("bdod", "0-5cm")]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def query_sg(lat, lon, tries=3):
    q = "&".join(f"property={p}" for p, _ in PROPS)
    url = (f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
           f"lon={lon}&lat={lat}&{q}&depth=0-5cm&value=mean")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                o = json.loads(r.read().decode())
            out = {}
            for layer in o["properties"]["layers"]:
                v = layer["depths"][0]["values"].get("mean")
                f = layer["unit_measure"].get("d_factor", 1) or 1
                out[layer["name"]] = None if v is None else v / f
            return out
        except Exception:
            time.sleep(0.8 + k)
    return None


def build_features(df):
    rows = []
    t0 = time.time()
    def work(i):
        r = df.iloc[i]
        s = query_sg(r["lat"], r["lon"])
        rec = {"crop": r["crop"], "region": r["region"], "lat": r["lat"], "lon": r["lon"]}
        if s:
            for p, _ in PROPS:
                rec[p] = s.get(p)
        rows.append(rec)
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(work, range(len(df))))
    feat = pd.DataFrame(rows).dropna(subset=[p for p, _ in PROPS])
    log(f"features built: {len(feat)}/{len(df)} rows with all 6 props, {time.time()-t0:.0f}s")
    return feat


def majority_baseline(ytr, yte):
    vals, counts = np.unique(ytr, return_counts=True)
    maj = vals[np.argmax(counts)]
    return float(np.mean(np.asarray(yte) == maj))


def run_cell(tag, feat, train_mask, test_mask, classes):
    cols = [p for p, _ in PROPS]
    Xtr = feat.loc[train_mask, cols].to_numpy("float32")
    ytr_raw = feat.loc[train_mask, "crop"].to_numpy()
    Xte = feat.loc[test_mask, cols].to_numpy("float32")
    yte_raw = feat.loc[test_mask, "crop"].to_numpy()
    codes = {c: i for i, c in enumerate(classes)}
    ytr = np.array([codes[c] for c in ytr_raw])
    yte = np.array([codes[c] for c in yte_raw])
    t0 = time.time()
    clf = tr.make_classifier()
    clf.fit(Xtr, ytr)
    fit_s = time.time() - t0
    t0 = time.time()
    proba = tr.predict_class_proba(clf, Xte)
    pred_s = time.time() - t0
    rep = sm.classification_report(proba, yte, n_classes=len(classes))
    rep.update({
        "cell": f"real_R2_{tag}", "task": "real_crop_from_soil_transfer",
        "train_n": int(train_mask.sum()), "test_n": int(test_mask.sum()),
        "train_regions": sorted(feat.loc[train_mask, "region"].unique().tolist()),
        "test_regions": sorted(feat.loc[test_mask, "region"].unique().tolist()),
        "majority_baseline": round(majority_baseline(ytr, yte), 4),
        "fit_s": round(fit_s, 2), "predict_s": round(pred_s, 2),
        "soil_source": "ISRIC SoilGrids v2.0 REST API (0-5cm, mean)",
        "label_source": "ESA WorldCereal point parquet (satellite-derived real labels)",
    })
    rep.update(tr.describe_version())
    with open(os.path.join(RES, rep["cell"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    log(f"{tag}: acc={rep['accuracy']:.3f} f1={rep['macro_f1']:.3f} "
        f"majority={rep['majority_baseline']:.3f} (train {train_mask.sum()} / test {test_mask.sum()})")
    return rep


def main():
    csv = os.path.join(ROOT, "data", "samples", "worldcereal_real_sample.csv")
    df = pd.read_csv(csv)
    log(f"loaded {len(df)} real crop-label points: {df['region'].value_counts().to_dict()}")
    feat = build_features(df)
    feat.to_csv(os.path.join(ROOT, "data", "raw", "real_R2_features.csv"), index=False)

    classes = sorted(feat["crop"].unique())
    rng = np.random.RandomState(42)
    n = len(feat)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    ridx = np.zeros(n, bool); ridx[perm[:cut]] = True
    is_us = (feat["region"] == "US").to_numpy()

    cells = {
        "random_mixed": (ridx, ~ridx),
        "trainUS_testAfrica": (is_us, ~is_us),
        "trainAfrica_testUS": (~is_us, is_us),
    }
    out = {}
    for tag, (trm, tem) in cells.items():
        if trm.sum() < 20 or tem.sum() < 20:
            log(f"skip {tag} (too few rows)"); continue
        out[tag] = run_cell(tag, feat, trm, tem, classes)

    summary = {
        "experiment": "R2_real_soil_to_real_crop_transfer",
        "n_features": len(PROPS), "classes": classes,
        "n_points_total": int(n), "n_points_with_soil": int(len(feat)),
        "soil_coverage": round(len(feat) / n, 3),
        "cells": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                      "majority_baseline": v["majority_baseline"]} for k, v in out.items()},
    }
    with open(os.path.join(RES, "real_R2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    log("R2 done: " + json.dumps(summary["cells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
