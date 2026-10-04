#!/usr/bin/env python3
"""Draw the R2 transfer-gap figure FROM THE RESULT JSONs ONLY.

    python3 scripts/make_fig_r2.py

Bars: accuracy per R2 cell, with the majority baseline as a marker, so the gap between
"random split looks great" and "does not transfer" is the visible message.
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

ORDER = ["random_mixed", "withinUS", "withinAfrica", "trainUS_testAfrica", "trainAfrica_testUS"]
LABEL = {
    "random_mixed": "random\nmixed split",
    "withinUS": "within US",
    "withinAfrica": "within Africa",
    "trainUS_testAfrica": "US -> Africa\n(transfer)",
    "trainAfrica_testUS": "Africa -> US\n(transfer)",
}


def main():
    cells = {}
    for f in glob.glob(os.path.join(RES, "real_R2_*.json")):
        d = json.load(open(f))
        if d.get("cell", "").startswith("real_R2_"):
            cells[d["cell"].replace("real_R2_", "")] = d
    got = [c for c in ORDER if c in cells]
    acc = [cells[c]["accuracy"] for c in got]
    maj = [cells[c]["majority_baseline"] for c in got]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(got))
    colors = ["#2b7a3d" if a - m > 0.1 else "#b0413e" for a, m in zip(acc, maj)]
    ax.bar(x, acc, color=colors, alpha=0.85, label="TabPFN-3.5 accuracy")
    ax.plot(x, maj, "k_", markersize=26, markeredgewidth=3, label="majority baseline")
    for xi, (a, m) in enumerate(zip(acc, maj)):
        ax.annotate(f"{a:.3f}", (xi, a), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)
        ax.annotate(f"{m:.3f}", (xi, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="k")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in got], fontsize=9)
    ax.set_ylabel("accuracy (real soil -> real crop)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Real soil→crop: random split looks strong, but it does not transfer across continents")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG, "real_transfer_gap.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)
    man = {c: {"accuracy": cells[c]["accuracy"], "majority_baseline": cells[c]["majority_baseline"],
               "n_test": cells[c]["n_test"]} for c in got}
    with open(os.path.join(FIG, "manifest_r2.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("wrote figures/manifest_r2.json")


if __name__ == "__main__":
    main()
