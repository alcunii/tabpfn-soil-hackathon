#!/usr/bin/env python3
"""R2-fit — REAL soil -> REAL crop with a geographic transfer test.

Reads the pre-built real feature table (data/raw/real_R2_features.csv: real WorldCereal crop
labels + real SoilGrids soil, built by build_real_R2_features.py) and fits TabPFN-3.5 on it.

Cells:
  * random_mixed        : 70/30 random (the optimistic number)
  * trainUS_testAfrica  : cross-region transfer
  * trainAfrica_testUS  : cross-region transfer (other direction)
  * withinUS / withinAfrica : region-internal 70/30 (is the transfer gap real?)

Every cell reports the MAJORITY-CLASS baseline next to accuracy, so "skill" is never overstated.

RUNS ON THE WORKSTATION. Writes results/real_R2_*.json.
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from soilhack import metrics as sm  # noqa: E402
from soilhack import tabpfn_runner as tr  # noqa: E402

PROPS = ["phh2o", "soc", "clay", "nitrogen", "cec", "bdod"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def majority(ytr, yte):
    v, c = np.unique(ytr, return_counts=True)
    return float(np.mean(np.asarray(yte) == v[np.argmax(c)]))


def fit_cell(tag, feat, trm, tem, classes):
    codes = {c: i for i, c in enumerate(classes)}
    Xtr = feat.loc[trm, PROPS].to_numpy("float32")
    ytr = np.array([codes[c] for c in feat.loc[trm, "crop"]])
    Xte = feat.loc[tem, PROPS].to_numpy("float32")
    yte = np.array([codes[c] for c in feat.loc[tem, "crop"]])
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
        "train_n": int(trm.sum()), "test_n": int(tem.sum()),
        "train_regions": sorted(feat.loc[trm, "region"].unique().tolist()),
        "test_regions": sorted(feat.loc[tem, "region"].unique().tolist()),
        "train_crop_counts": feat.loc[trm, "crop"].value_counts().to_dict(),
        "test_crop_counts": feat.loc[tem, "crop"].value_counts().to_dict(),
        "majority_baseline": round(majority(ytr, yte), 4),
        "fit_s": round(fit_s, 2), "predict_s": round(pred_s, 2),
        "soil_source": "ISRIC SoilGrids v2.0 (0-5cm mean)",
        "label_source": "ESA WorldCereal point parquet (satellite-derived)",
    })
    rep.update(tr.describe_version())
    with open(os.path.join(RES, rep["cell"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    log(f"{tag}: acc={rep['accuracy']:.3f} f1={rep['macro_f1']:.3f} maj={rep['majority_baseline']:.3f} "
        f"(train {trm.sum()}/{sorted(feat.loc[trm,'region'].unique())} "
        f"test {tem.sum()}/{sorted(feat.loc[tem,'region'].unique())})")
    return rep


def main():
    csv = os.path.join(ROOT, "data", "raw", "real_R2_features.csv")
    df = pd.read_csv(csv)
    df = df[df.get("soil_ok", 1) == 1].dropna(subset=PROPS)
    log(f"real feature table: {len(df)} rows with full soil | {df['region'].value_counts().to_dict()}")
    classes = sorted(df["crop"].unique())
    rng = np.random.RandomState(42)
    n = len(df)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    ridx = np.zeros(n, bool); ridx[perm[:cut]] = True
    is_us = (df["region"] == "US").to_numpy()
    is_af = (df["region"] == "Africa").to_numpy()

    # within-region 70/30
    def split_within(mask):
        idx = np.where(mask)[0]
        p = rng.permutation(idx)
        c = int(0.7 * len(p))
        trm = np.zeros(n, bool); tem = np.zeros(n, bool)
        trm[p[:c]] = True; tem[p[c:]] = True
        return trm, tem

    us_tr, us_te = split_within(is_us)
    af_tr, af_te = split_within(is_af)

    cells = {
        "random_mixed": (ridx, ~ridx),
        "trainUS_testAfrica": (is_us, is_af),
        "trainAfrica_testUS": (is_af, is_us),
        "withinUS": (us_tr, us_te),
        "withinAfrica": (af_tr, af_te),
    }
    out = {}
    for tag, (trm, tem) in cells.items():
        if trm.sum() < 30 or tem.sum() < 30:
            log(f"skip {tag} (rows {trm.sum()}/{tem.sum()})"); continue
        try:
            out[tag] = fit_cell(tag, df, trm, tem, classes)
        except Exception as e:  # noqa: BLE001
            log(f"{tag} FAILED {type(e).__name__}: {str(e)[:200]}")

    summary = {
        "experiment": "R2_real_soil_to_real_crop_transfer",
        "n_features": len(PROPS), "classes": classes, "n_rows": int(n),
        "cells": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                      "majority_baseline": v["majority_baseline"], "n_test": v["n_test"]}
                  for k, v in out.items()},
        "reading": "compare each cell's accuracy against its majority baseline; the "
                   "trainUS_testAfrica vs withinUS difference is the transfer gap.",
    }
    with open(os.path.join(RES, "real_R2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    log("SUMMARY " + json.dumps(summary["cells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
