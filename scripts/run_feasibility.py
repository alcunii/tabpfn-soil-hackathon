#!/usr/bin/env python3
"""Feasibility experiments for the soilhack hackathon ideas.

RUNS ON THE GPU WORKSTATION (tabpfn 9.0.0, TabPFN-3.5, an NVIDIA GPU).
Writes one JSON per cell into results/ and a progress line per cell.

    python scripts\\run_feasibility.py core     # the headline cells
    python scripts\\run_feasibility.py sweep    # context-size sweep (few-shot story)

Grids
-----
core:
  E1  crop recommendation classification (7 soil/climate features -> 22 crops)
  E2  OSSL SOC regression from vis-NIR spectra (1051 SNV bands -> SOC)
sweep:
  E1s / E2s  the same tasks at several context sizes -> the TabPFN few-shot curve

Every cell records the resolved TabPFN version + the context size next to the metric.
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # ...\soilhack
sys.path.insert(0, os.path.join(ROOT, "src"))
RESDIR = os.path.join(ROOT, "results")
os.makedirs(RESDIR, exist_ok=True)

import numpy as np  # noqa: E402

from soilhack import data as sdata  # noqa: E402
from soilhack import metrics as sm  # noqa: E402
from soilhack import tabpfn_runner as tr  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(cell, obj):
    path = os.path.join(RESDIR, cell + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    log(f"saved {cell}.json")


def write_env_probe():
    info = tr.describe_version()
    info["host_note"] = "GPU workstation, reached via reverse-tunnel job runner"
    save("env_probe", info)
    return info


# ------------------------------------------------------------------ E1 classification
def cell_e1_croprec(tag, n_context=None, seed=42):
    log(f"E1 {tag}: loading crop recommendation")
    X, y, classes = sdata.load_crop_recommendation()
    X = X.to_numpy("float32")
    n, n_classes = len(y), len(classes)
    tr_idx, te_idx = sdata.train_test_split_idx(n, frac=0.7, seed=seed)
    if n_context is not None and n_context < len(tr_idx):
        rng = np.random.RandomState(seed)
        tr_idx = np.sort(rng.choice(tr_idx, size=n_context, replace=False))
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    log(f"E1 {tag}: ctx={len(tr_idx)} test={len(te_idx)} classes={n_classes}")
    t0 = time.time()
    clf = tr.make_classifier()
    clf.fit(Xtr, ytr)
    fit_s = time.time() - t0
    t0 = time.time()
    proba = tr.predict_class_proba(clf, Xte)
    pred_s = time.time() - t0
    rep = sm.classification_report(proba, yte, n_classes=n_classes)
    rep.update({
        "cell": f"eas_e1_{tag}",
        "task": "crop_recommendation_classification",
        "dataset": "kaggle atharvaingle/crop-recommendation-dataset (Apache-2.0)",
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "n_context": int(len(tr_idx)),
        "n_train_total": int(n * 0.7),
        "fit_s": round(fit_s, 2),
        "predict_s": round(pred_s, 2),
        "seed": seed,
    })
    rep.update(tr.describe_version())
    save(f"feas_e1_{tag}", rep)
    return rep


# ------------------------------------------------------------------ E2 regression
def cell_e2_ossl_soc(tag, n_context=None, seed=42):
    log(f"E2 {tag}: loading OSSL SOC spectra sample")
    X, y, meta = sdata.load_ossl_soc_sample()
    n = len(y)
    tr_idx, te_idx = sdata.train_test_split_idx(n, frac=0.7, seed=seed)
    if n_context is not None and n_context < len(tr_idx):
        rng = np.random.RandomState(seed)
        tr_idx = np.sort(rng.choice(tr_idx, size=n_context, replace=False))
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    log(f"E2 {tag}: ctx={len(tr_idx)} test={len(te_idx)} bands={X.shape[1]}")
    t0 = time.time()
    reg = tr.make_regressor()
    reg.fit(Xtr, ytr)
    fit_s = time.time() - t0
    t0 = time.time()
    pred = tr.predict_regression(reg, Xte)
    pred_s = time.time() - t0
    rep = sm.regression_report(yte, pred)
    # constant baseline = collapse detector (mandatory)
    const = np.full_like(yte, ytr.mean(), dtype=float)
    rep_const = sm.regression_report(yte, const)
    rep.update({
        "cell": f"feas_e2_{tag}",
        "task": "ossl_soc_regression_from_spectra",
        "dataset": "OSSL v1.2 vis-NIR (open), 400-spectrum stratified sample",
        "n_features": X.shape[1],
        "n_context": int(len(tr_idx)),
        "n_train_total": int(n * 0.7),
        "fit_s": round(fit_s, 2),
        "predict_s": round(pred_s, 2),
        "seed": seed,
        "constant_baseline": {"rmse": rep_const["rmse"], "r2": rep_const["r2"], "rpd": rep_const["rpd"]},
    })
    rep.update(tr.describe_version())
    save(f"feas_e2_{tag}", rep)
    return rep


# ------------------------------------------------------------------ grids
def grid_core():
    write_env_probe()
    cell_e1_croprec("full")
    cell_e2_ossl_soc("full")
    log("core grid done")


def grid_sweep():
    write_env_probe()
    for k in (100, 500, 1000, None):
        tag = "all" if k is None else f"k{k}"
        try:
            cell_e1_croprec(f"sweep_{tag}", n_context=k)
        except Exception as e:  # noqa: BLE001
            log(f"E1 sweep {tag} failed: {e}")
        try:
            cell_e2_ossl_soc(f"sweep_{tag}", n_context=k)
        except Exception as e:  # noqa: BLE001
            log(f"E2 sweep {tag} failed: {e}")
    log("sweep grid done")


def main():
    grid = sys.argv[1] if len(sys.argv) > 1 else "core"
    log(f"--- run_feasibility grid={grid} python={sys.executable} ---")
    log(f"    tabpfn probe: {tr.describe_version()}")
    if grid == "core":
        grid_core()
    elif grid == "sweep":
        grid_sweep()
    else:
        log(f"unknown grid {grid}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
