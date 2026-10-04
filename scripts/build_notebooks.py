#!/usr/bin/env python3
"""Build the teaching notebooks from source (never hand-edit a .ipynb).

    python3 scripts/build_notebooks.py        # writes notebooks/*.ipynb

Cells are Windows-safe: relative paths / repo-root detection, no /tmp, no shell.
Every code cell is a top-level cell with its own imports so a fresh kernel per cell works.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NBDIR = os.path.join(ROOT, "notebooks")
os.makedirs(NBDIR, exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


BOOT = (
    "import os, sys\n"
    "ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
    "if not os.path.isdir(os.path.join(ROOT, 'src')):\n"
    "    ROOT = os.getcwd()\n"
    "sys.path.insert(0, os.path.join(ROOT, 'src'))\n"
    "print('repo root:', ROOT)\n"
)

NB1 = [
    md("# soilhack — feasibility walkthrough (TabPFN-3.5)\n\n"
       "This notebook **runs on the GPU workstation** and reproduces the two headline\n"
       "feasibility results from `RESULTS.md`. No training loop — one `fit()` per task.\n\n"
       "Run each cell in order."),
    code(BOOT),
    md("## 0. What model are we actually using?\n\n"
       "We **name the checkpoint explicitly** (V3.5). A bare constructor can silently load a\n"
       "different generation, which would move every number."),
    code("from soilhack import tabpfn_runner as tr\nprint(tr.describe_version())"),
    md("## 1. Crop recommendation (classification)\n\n"
       "7 soil/climate features -> 22 crops. Watch the accuracy at a **100-row context** —\n"
       "the few-shot story."),
    code(
        "import numpy as np\n"
        "from soilhack import data as sdata, metrics as sm\n"
        "X, y, classes = sdata.load_crop_recommendation()\n"
        "X = X.to_numpy('float32')\n"
        "tr_i, te_i = sdata.train_test_split_idx(len(y), 0.7, 42)\n"
        "rng = np.random.RandomState(42); small = np.sort(rng.choice(tr_i, 100, replace=False))\n"
        "clf = tr.make_classifier(); clf.fit(X[small], y[small])\n"
        "proba = tr.predict_class_proba(clf, X[te_i])\n"
        "print('classes:', len(classes), '| ctx=100 | test=', len(te_i))\n"
        "print(sm.classification_report(proba, y[te_i], n_classes=len(classes)))"
    ),
    md("## 2. Soil organic carbon from vis-NIR spectra (regression)\n\n"
       "1,051 SNV bands -> SOC. Compare to the constant baseline (the collapse detector)."),
    code(
        "from soilhack import data as sdata, metrics as sm\n"
        "Xs, ys, meta = sdata.load_ossl_soc_sample()\n"
        "tr_i, te_i = sdata.train_test_split_idx(len(ys), 0.7, 42)\n"
        "reg = tr.make_regressor(); reg.fit(Xs[tr_i], ys[tr_i])\n"
        "pred = tr.predict_regression(reg, Xs[te_i])\n"
        "rep = sm.regression_report(ys[te_i], pred)\n"
        "const = sm.regression_report(ys[te_i], np.full(len(te_i), ys[tr_i].mean()))\n"
        "print('TabPFN-3.5:', {k: round(rep[k], 3) for k in ('r2','rmse','rpd','rpiq')})\n"
        "print('constant  :', {k: round(const[k], 3) for k in ('r2','rmse','rpd')})"
    ),
    md("## 3. Reading rules\n\n"
       "1. A skill number is meaningless without its **context size** and **test set**.\n"
       "2. Always report the **constant baseline** with a regression number.\n"
       "3. Classification: accuracy **+ macro-F1 + ECE**, never accuracy alone."),
]

NB2 = [
    md("# soilhack — the idea menu, in code\n\n"
       "A short, runnable map of the four candidate ideas and the evidence each rests on.\n"
       "Read `docs/IDEAS.md` alongside this."),
    code(BOOT),
    md("## The four ideas\n"
       "1. **SoilSense** — few-shot soil-property prediction from cheap spectra *(formalize a problem)*\n"
       "2. **NutriRx** — soil test -> fertilizer & lime prescription *(app)*\n"
       "3. **SoilAdvisor** — an agent that predicts, forecasts, acts *(agent)*\n"
       "4. **FieldAlert** — crop-stress / yield early warning *(hard problem / climate)*\n"),
    md("## Evidence available right now\n\n"
       "Load every result JSON and print the headline number per cell. This is the honest,\n"
       "measured state of the project."),
    code(
        "import json, glob, os\n"
        "ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
        "if not os.path.isdir(os.path.join(ROOT, 'results')):\n"
        "    ROOT = os.getcwd()\n"
        "rows = []\n"
        "for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'feas_e*.json'))):\n"
        "    d = json.load(open(f))\n"
        "    if d['task'].startswith('crop'):\n"
        "        rows.append((d['cell'], d['n_context'], 'acc=%.4f' % d['accuracy'], 'ece=%.4f' % d['ece']))\n"
        "    else:\n"
        "        rows.append((d['cell'], d['n_context'], 'r2=%.3f' % d['r2'], 'rpd=%.2f(%s)' % (d['rpd'], d['rpd_label'])))\n"
        "for r in rows:\n"
        "    print(f'{r[0]:<26} ctx={r[1]:>5}  {r[2]:<12} {r[3]}')"
    ),
    md("## Which to build\n\n"
       "My read: **SoilSense** (measured depth) as the core + **NutriRx** (demo-friendly) as the\n"
       "app layer. See `docs/IDEAS.md` §6. Nothing is chosen yet."),
]


def write_nb(path, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                       "name": "python3"},
                                        "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path, flush=True)


write_nb(os.path.join(NBDIR, "01_feasibility_walkthrough.ipynb"), NB1)
write_nb(os.path.join(NBDIR, "02_idea_menu.ipynb"), NB2)
