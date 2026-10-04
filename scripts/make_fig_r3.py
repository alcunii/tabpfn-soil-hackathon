#!/usr/bin/env python3
"""Draw the few-shot learning curve FROM THE RESULT JSONs ONLY.

    python3 scripts/make_fig_r3.py

Accuracy vs number of local labelled rows (k), spatially blocked, for soil+climate and
climate-only, with the majority baseline and the k=0 transfer point.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(prefix):
    out = {}
    for f in glob.glob(os.path.join(RES, prefix + "*.json")):
        d = json.load(open(f))
        out[d.get("cell", os.path.basename(f))] = d
    return out


def main():
    r3b = load("real_R3b_")
    ks, full, clim = [], [], []
    for k in (0, 5, 10, 20, 50, 100):
        c = f"real_R3b_fewshot_blocked_k{k}"
        if c in r3b:
            ks.append(k); full.append(r3b[c]["accuracy"])
        cc = f"real_R3b_climateonly_k{k}"
        if cc in r3b:
            clim.append((k, r3b[cc]["accuracy"]))
    maj = r3b.get("real_R3b_fewshot_blocked_k0", {}).get("majority_baseline")
    lr = r3b.get("real_R3b_baseline_lr_k20", {}).get("accuracy")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ks, full, "o-", color="#1f4e79", lw=2, label="TabPFN-3.5 (soil + climate), blocked")
    if clim:
        xc = [c[0] for c in clim]; yc = [c[1] for c in clim]
        ax.plot(xc, yc, "^--", color="#2b7a3d", lw=1.5, alpha=0.8, label="climate only, blocked")
    if maj is not None:
        ax.axhline(maj, ls=":", color="grey", lw=1.2, label=f"majority baseline = {maj:.2f}")
    if lr is not None:
        ax.scatter([20], [lr], marker="s", color="#b0413e", zorder=5, label=f"logistic reg @ k=20 = {lr:.3f}")
    for x, y in zip(ks, full):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("local labelled rows added for the target region (k)")
    ax.set_ylabel("accuracy (real crop, blocked regions)")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Few-shot adaptation: ~5 local points solve it — carried mostly by climate")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG, "fewshot_curve.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)
    with open(os.path.join(FIG, "manifest_r3.json"), "w") as f:
        json.dump({"k": ks, "soilclim_blocked": full, "climate_only": clim,
                   "majority": maj, "lr_k20": lr}, f, indent=1)
    print("wrote figures/manifest_r3.json")


if __name__ == "__main__":
    main()
