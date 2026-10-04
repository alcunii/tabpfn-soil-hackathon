#!/usr/bin/env python3
"""Draw the feasibility figures FROM THE RESULT JSONs ONLY (never hand-typed numbers).

    python3 scripts/make_figures.py        # on the VPS; writes figures/*.png

Two panels: (1) crop-recommendation accuracy vs context size; (2) SOC R2 / RPD vs context size.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(pattern):
    out = {}
    for f in glob.glob(os.path.join(RES, pattern)):
        d = json.load(open(f))
        out[d["cell"]] = d
    return out


def ctx_of(d):
    return d["n_context"]


def main():
    e1 = load("feas_e1_*.json")
    e2 = load("feas_e2_*.json")

    # de-dupe: prefer the 'sweep_*' cells for the curve, keep 'full' for the headline
    def curve(cells, key):
        pts = []
        for c, d in cells.items():
            if "sweep" not in c:
                continue
            pts.append((ctx_of(d), d[key], c))
        pts.sort()
        return pts

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- E1 classification
    ax = axes[0]
    pts = curve(e1, "accuracy")
    if pts:
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax.plot(x, y, "o-", color="#2b7a3d", lw=2, label="TabPFN-3.5 accuracy")
        for xi, yi, _ in pts:
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.axhline(1.0 / 22, ls="--", color="grey", lw=1, label="chance (1/22 = 0.045)")
        ax.set_xlabel("context rows (labelled)")
        ax.set_ylabel("accuracy")
        ax.set_title("E1 crop recommendation (22 classes)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- E2 regression
    ax = axes[1]
    pts = curve(e2, "r2")
    if pts:
        x = [p[0] for p in pts]
        r2 = [p[1] for p in pts]
        ax.plot(x, r2, "o-", color="#1f4e79", lw=2, label="TabPFN-3.5 R²")
        for xi, yi, _ in pts:
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        # constant baseline
        const_r2 = next((d["constant_baseline"]["r2"] for d in e2.values()), None)
        if const_r2 is not None:
            ax.axhline(const_r2, ls="--", color="grey", lw=1, label=f"constant baseline R²={const_r2:.2f}")
        ax.set_xlabel("context rows (labelled)")
        ax.set_ylabel("R² (SOC)")
        ax.set_title("E2 soil organic carbon from vis-NIR spectra")
        ax.set_ylim(-0.1, 1.0)
        ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("TabPFN-3.5 feasibility — few labelled rows still predict well (GPU workstation, GPU)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "feasibility.png")
    fig.savefig(out, dpi=140)
    print("wrote", out, flush=True)

    # a manifest so a reader can verify each figure's numbers
    man = {
        "feasibility.png": {
            "e1_accuracy_by_context": {str(ctx_of(d)): d["accuracy"] for d in e1.values() if "sweep" in d["cell"]},
            "e2_r2_by_context": {str(ctx_of(d)): d["r2"] for d in e2.values() if "sweep" in d["cell"]},
            "e2_constant_r2": next((d["constant_baseline"]["r2"] for d in e2.values()), None),
        }
    }
    with open(os.path.join(FIG, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("wrote figures/manifest.json", flush=True)


if __name__ == "__main__":
    main()
