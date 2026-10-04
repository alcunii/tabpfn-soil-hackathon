#!/usr/bin/env python3
"""R3 — does CLIMATE fix the transfer gap? And how little local data does a new region need?

Adds climate covariates (Open-Meteo/ERA5) to the real soil features and re-runs the R2
transfer protocol, then measures a FEW-SHOT learning curve: adapt to a target region using
only k local labelled rows (k = 5,10,20,50,100,200). This is the "accurate one" evidence:
climate should help, and a handful of local rows should be enough.

Inputs : data/raw/real_R2_features.csv  (soil, real labels)  -- left-joined with
         data/raw/real_climate_features.csv (on lat,lon)
RUNS ON THE WORKSTATION. Writes results/real_R3_*.json.
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

SOIL = ["phh2o", "soc", "clay", "nitrogen", "cec", "bdod"]
CLIM = ["tmean", "trange", "prec_annual", "prec_cv", "gdd5", "frost_days", "aridity"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def majority(ytr, yte):
    v, c = np.unique(ytr, return_counts=True)
    return float(np.mean(np.asarray(yte) == v[np.argmax(c)]))


def fit_cell(tag, df, cols, trm, tem, classes):
    codes = {c: i for i, c in enumerate(classes)}
    Xtr = df.loc[trm, cols].to_numpy("float32")
    ytr = np.array([codes[c] for c in df.loc[trm, "crop"]])
    Xte = df.loc[tem, cols].to_numpy("float32")
    yte = np.array([codes[c] for c in df.loc[tem, "crop"]])
    clf = tr.make_classifier()
    clf.fit(Xtr, ytr)
    proba = tr.predict_class_proba(clf, Xte)
    rep = sm.classification_report(proba, yte, n_classes=len(classes))
    rep.update({
        "cell": f"real_R3_{tag}", "n_features": len(cols), "feature_set": "+".join(
            ["soil"] + (["clim"] if any(c in CLIM for c in cols) else [])),
        "train_n": int(trm.sum()), "test_n": int(tem.sum()),
        "train_regions": sorted(df.loc[trm, "region"].unique().tolist()),
        "test_regions": sorted(df.loc[tem, "region"].unique().tolist()),
        "majority_baseline": round(majority(ytr, yte), 4),
    })
    rep.update(tr.describe_version())
    with open(os.path.join(RES, rep["cell"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    log(f"{tag}: acc={rep['accuracy']:.3f} f1={rep['macro_f1']:.3f} maj={rep['majority_baseline']:.3f} "
        f"({rep['feature_set']}, train {int(trm.sum())}/test {int(tem.sum())})")
    return rep


def main():
    fsoil = os.path.join(ROOT, "data", "raw", "real_R2_features.csv")
    fclim = os.path.join(ROOT, "data", "raw", "real_climate_features.csv")
    soil = pd.read_csv(fsoil)
    soil = soil[soil.get("soil_ok", 1) == 1].dropna(subset=SOIL)
    clim = pd.read_csv(fclim)
    soil["k"] = soil["lat"].round(3).astype(str) + "|" + soil["lon"].round(3).astype(str)
    clim["k"] = clim["lat"].round(3).astype(str) + "|" + clim["lon"].round(3).astype(str)
    df = soil.merge(clim.drop(columns=["lat", "lon"]), on="k", how="left")
    n_clim = int(df[CLIM].notna().all(axis=1).sum())
    log(f"merged: {len(df)} rows, {n_clim} with full climate | {df['region'].value_counts().to_dict()}")
    classes = sorted(df["crop"].unique())

    out = {}
    is_us = (df["region"] == "US").to_numpy()
    is_af = (df["region"] == "Africa").to_numpy()
    rng = np.random.RandomState(42)

    # ---- 1) soil vs soil+climate on the same rows (both regions, random split)
    both = df[df[CLIM].notna().all(axis=1)].reset_index(drop=True)
    p = rng.permutation(len(both)); cut = int(0.7 * len(both))
    trm = np.zeros(len(both), bool); trm[p[:cut]] = True; tem = ~trm
    for tag, cols in [("soilonly_randomsplit", SOIL), ("soilclim_randomsplit", SOIL + CLIM)]:
        try:
            out[tag] = fit_cell(tag, both, cols, trm, tem, classes)
        except Exception as e:  # noqa: BLE001
            log(f"{tag} FAILED {type(e).__name__}: {str(e)[:200]}")

    # ---- 2) transfer with climate: train US -> test Africa, and reverse
    us = df[is_us & df[CLIM].notna().all(axis=1)].index
    af = df[is_af & df[CLIM].notna().all(axis=1)].index
    for tag, (trm_, tem_), cols in [
        ("soilclim_US2AF", (is_us, is_af), SOIL + CLIM),
        ("soilclim_AF2US", (is_af, is_us), SOIL + CLIM),
    ]:
        # restrict to rows with climate
        a = np.zeros(len(df), bool); a[us] = True
        b = np.zeros(len(df), bool); b[af] = True
        trm2 = a if tag.endswith("US2AF") else b
        tem2 = b if tag.endswith("US2AF") else a
        try:
            out[tag] = fit_cell(tag, df, cols, trm2, tem2, classes)
        except Exception as e:  # noqa: BLE001
            log(f"{tag} FAILED {type(e).__name__}: {str(e)[:200]}")

    # ---- 3) few-shot: adapt to Africa from US + k local African rows
    af_all = np.where(is_af & df[CLIM].notna().all(axis=1))[0]
    rng.shuffle(af_all)
    af_train_pool, af_test = af_all[:120], af_all[120:]
    for k in (0, 5, 10, 20, 50, 100):
        local = af_train_pool[:k]
        trm = np.zeros(len(df), bool); trm[us] = True; trm[local] = True
        tem = np.zeros(len(df), bool); tem[af_test] = True
        if tem.sum() < 20:
            continue
        try:
            out[f"fewshot_AF_k{k}"] = fit_cell(f"fewshot_AF_k{k}", df, SOIL + CLIM, trm, tem, classes)
        except Exception as e:  # noqa: BLE001
            log(f"fewshot k{k} FAILED {type(e).__name__}: {str(e)[:200]}")

    summary = {
        "experiment": "R3_climate_and_fewshot",
        "n_rows": int(len(df)), "n_with_climate": int(n_clim),
        "classes": classes,
        "soil_features": SOIL, "climate_features": CLIM,
        "cells": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                      "majority_baseline": v["majority_baseline"], "n_test": v["n_test"]}
                  for k, v in out.items()},
    }
    with open(os.path.join(RES, "real_R3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    log("R3 SUMMARY " + json.dumps(summary["cells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
