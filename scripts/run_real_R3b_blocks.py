#!/usr/bin/env python3
"""R3b — FEW-SHOT with SPATIAL BLOCKING (the honesty check on R3's few-shot result).

R3's few-shot hit 1.000 accuracy at k=10, which is suspicious: climate features are constant
within a 0.25-degree cell, and soil is smooth over ~25 km, so a "local" training row drawn from
the same cell as a test row leaks location. This script removes that leak:

  * Africa points are grouped into coarse spatial blocks (0.5-degree cells).
  * Blocks are split by block into a POOL half and a TEST half (never the same block).
  * test rows come only from TEST blocks; the k local training rows come only from POOL blocks.

It also quantifies the leak in the unblocked setting (fraction of test rows sharing a block with
a local training row). If blocking drops the few-shot accuracy, the R3 number was inflated.

RUNS ON THE WORKSTATION. Writes results/real_R3b_*.json.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from soilhack import metrics as sm  # noqa: E402
from soilhack import tabpfn_runner as tr  # noqa: E402

SOIL = ["phh2o", "soc", "clay", "nitrogen", "cec", "bdod"]
CLIM = ["tmean", "trange", "prec_annual", "prec_cv", "gdd5", "frost_days", "aridity"]
BLOCK = 0.5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def block_of(lat, lon, res=BLOCK):
    return f"{round(lat / res):d}|{round(lon / res):d}"


def majority(ytr, yte):
    v, c = np.unique(ytr, return_counts=True)
    return float(np.mean(np.asarray(yte) == v[np.argmax(c)]))


def baseline_lr(df, cols, trm, tem):
    codes = {c: i for i, c in enumerate(sorted(df["crop"].unique()))}
    Xtr = StandardScaler().fit_transform(df.loc[trm, cols].to_numpy("float32"))
    ytr = np.array([codes[c] for c in df.loc[trm, "crop"]])
    Xte = StandardScaler().fit(df.loc[trm, cols].to_numpy("float32")).transform(df.loc[tem, cols].to_numpy("float32"))
    yte = np.array([codes[c] for c in df.loc[tem, "crop"]])
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    p = lr.predict(Xte)
    return {"accuracy": float(np.mean(p == yte)), "majority_baseline": round(majority(ytr, yte), 4),
            "n_test": int(tem.sum())}


def fit_cell(tag, df, cols, trm, tem, classes, note):
    codes = {c: i for i, c in enumerate(classes)}
    Xtr = df.loc[trm, cols].to_numpy("float32")
    ytr = np.array([codes[c] for c in df.loc[trm, "crop"]])
    Xte = df.loc[tem, cols].to_numpy("float32")
    yte = np.array([codes[c] for c in df.loc[tem, "crop"]])
    clf = tr.make_classifier(); clf.fit(Xtr, ytr)
    proba = tr.predict_class_proba(clf, Xte)
    rep = sm.classification_report(proba, yte, n_classes=len(classes))
    rep.update({
        "cell": f"real_R3b_{tag}", "note": note,
        "train_n": int(trm.sum()), "test_n": int(tem.sum()),
        "majority_baseline": round(majority(ytr, yte), 4),
        "train_blocks": sorted(df.loc[trm, "block"].unique().tolist())[:5],
        "test_blocks": sorted(df.loc[tem, "block"].unique().tolist())[:5],
    })
    rep.update(tr.describe_version())
    with open(os.path.join(RES, rep["cell"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    log(f"{tag}: acc={rep['accuracy']:.3f} f1={rep['macro_f1']:.3f} maj={rep['majority_baseline']:.3f} "
        f"(train {int(trm.sum())}/test {int(tem.sum())})")
    return rep


def main():
    soil = pd.read_csv(os.path.join(ROOT, "data", "raw", "real_R2_features.csv"))
    soil = soil[soil.get("soil_ok", 1) == 1].dropna(subset=SOIL)
    clim = pd.read_csv(os.path.join(ROOT, "data", "raw", "real_climate_features.csv"))
    soil["k"] = soil["lat"].round(3).astype(str) + "|" + soil["lon"].round(3).astype(str)
    clim["k"] = clim["lat"].round(3).astype(str) + "|" + clim["lon"].round(3).astype(str)
    df = soil.merge(clim.drop(columns=["lat", "lon"]), on="k", how="left").reset_index(drop=True)
    df = df.dropna(subset=CLIM).reset_index(drop=True)
    df["block"] = [block_of(a, b) for a, b in zip(df["lat"], df["lon"])]
    classes = sorted(df["crop"].unique())
    log(f"rows {len(df)} | US {(df.region=='US').sum()} Africa {(df.region=='Africa').sum()} | "
        f"{df['block'].nunique()} blocks")

    is_us = (df["region"] == "US").to_numpy()
    is_af = (df["region"] == "Africa").to_numpy()
    rng = np.random.RandomState(7)

    # --- leak check: in the UNBLOCKED few-shot, how many test rows share a block with the
    #     local training rows? (k = 20 local rows drawn naively)
    af_idx = np.where(is_af)[0]
    rng.shuffle(af_idx)
    naive_local, af_test = af_idx[:120], af_idx[120:]
    local_blocks = set(df.loc[naive_local[:20], "block"])
    leaked = int(df.loc[af_test, "block"].isin(local_blocks).sum())
    log(f"LEAK CHECK: {leaked}/{len(af_test)} ({leaked/len(af_test):.1%}) test rows share a block "
        f"with the 20 local training rows (unblocked)")

    # --- SPATIALLY BLOCKED few-shot: split Africa BLOCKS (numpy-safe shuffle) into pool vs test
    af_blocks = np.array(df.loc[is_af, "block"].unique(), dtype=object).astype(str)
    rng.shuffle(af_blocks)
    pool_blocks = set(af_blocks[: len(af_blocks) // 2])
    in_pool_block = df["block"].isin(pool_blocks).to_numpy()
    pool_idx = np.where(is_af & in_pool_block)[0]
    test_idx = np.where(is_af & ~in_pool_block)[0]
    log(f"blocked split: pool blocks {len(pool_blocks)}, pool rows {len(pool_idx)}, test rows {len(test_idx)}")
    rng.shuffle(pool_idx)
    # independent baseline: logistic regression, k=20 local rows (no GPU, no TabPFN)
    b_lr = baseline_lr(df, SOIL + CLIM,
                       np.isin(np.arange(len(df)), np.concatenate([np.where(is_us)[0], pool_idx[:20]])),
                       np.isin(np.arange(len(df)), test_idx))
    log(f"BASELINE LR k20: acc={b_lr['accuracy']:.3f} maj={b_lr['majority_baseline']:.3f}")

    out = {"leak_check_unblocked": {"shared_block_test_rows": leaked, "n_test": int(len(af_test)),
                                    "fraction": round(leaked / max(len(af_test), 1), 4)},
           "baseline_lr_k20": b_lr}
    for k in (0, 5, 10, 20, 50, 100):
        local = pool_idx[:k]
        trm = np.zeros(len(df), bool); trm[is_us] = True; trm[local] = True
        tem = np.zeros(len(df), bool); tem[test_idx] = True
        try:
            out[f"fewshot_blocked_k{k}"] = fit_cell(
                f"fewshot_blocked_k{k}", df, SOIL + CLIM, trm, tem, classes,
                "spatially blocked (train-local and test in different 0.5deg cells)")
        except Exception as e:  # noqa: BLE001
            log(f"k{k} FAILED {type(e).__name__}: {str(e)[:160]}")

    # --- DIAGNOSTIC: is the 3-class task determined by CLIMATE alone?
    for tag, cols in [("climateonly_k0", CLIM), ("climateonly_k5", CLIM), ("soilonly_k5", SOIL)]:
        k = 0 if tag.endswith("k0") else 5
        local = pool_idx[:k]
        trm = np.zeros(len(df), bool); trm[is_us] = True; trm[local] = True
        tem = np.zeros(len(df), bool); tem[test_idx] = True
        try:
            out[tag] = fit_cell(tag, df, cols, trm, tem, classes, "DIAGNOSTIC feature-ablation, blocked")
        except Exception as e:  # noqa: BLE001
            log(f"{tag} FAILED {type(e).__name__}: {str(e)[:160]}")

    def cellentry(v):
        if isinstance(v, dict) and "accuracy" in v:
            return {"accuracy": v["accuracy"], "macro_f1": v.get("macro_f1"),
                    "majority_baseline": v.get("majority_baseline"), "n_test": v.get("n_test")}
        return v

    summary = {"experiment": "R3b_fewshot_spatial_blocks", "block_deg": BLOCK,
               "classes": classes, "n_rows": int(len(df)),
               "cells": {k: cellentry(v) for k, v in out.items()}}
    with open(os.path.join(RES, "real_R3b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    log("R3b SUMMARY " + json.dumps(summary["cells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
