# Competitive analysis — crop-recommendation projects, and the real gap

This is the deep look at what already exists, so we build *on top* of it rather than repeat it.
Two claims are made below; both were **verified by downloading the projects and reading their
code/data** on this workstation/VPS, not by assumption.

---

## 1. What exists (and how it scores)

### The deployed / popular open projects (code read directly)

| Project | What it is | Data used | Reported result |
|---|---|---|---|
| **Harvestify** (`github.com/Gladiator07/Harvestify`) | Deployed Flask web app: crop rec + fertilizer + disease | **Kaggle `atharvaingle/crop-recommendation-dataset`** (2,200 rows) | RF ~99% |
| **KRUTHIKTR/Crop-Recommendation-System-Using-Machine-Learning** | ML pipeline + notebook | same 2,200-row table (`crop_data1.csv`) | train 93.26% / val 92.53% |
| **AayushPanchal-04/Crop-Recommendation-System** | RF model + Streamlit | same table | test 99.55%, CV 99.43% |
| **Dozens of Kaggle notebooks** | same recipe | same table | 95–99.9% |

**Verified fact:** all of the above use the **identical** 2,200-row table — Harvestify's
`Data-processed/crop_recommendation.csv`, KRUTHIKTR's `crop_data1.csv` and AayushPanchal's
`crop_recommendation.csv` share the **same first row** (`90,42,43,20.88,82.00,6.50,202.94,rice`)
and the same 22 crops. They are one dataset re-published many times.

### Peer-reviewed work (same dataset, read the papers)

- *Advancing crop recommendation … explainable AI* (PMC12264067) — "dataset … sourced from
  Kaggle … 2,200 samples … 22 classes."
- *Interpretable deep learning for … crop recommendation* (Sci. Rep., s41598-025-26910-4) —
  trains on the same table, then **SMOTEs it to 22,000** synthetic rows.
- *Smart Farming: Crop Recommendation…* (Dahiphale 2025) — cites the same Kaggle dataset.
- A Zenodo "Crop Recommendation Dataset with SMOTE, VAE" explicitly warns *"synthetic data may
  introduce minor distributional deviations."*

## 2. The gap (two verified problems)

**Problem A — the benchmark is synthetic and self-referential.** The 2,200-row table was
*"built by augmenting datasets of rainfall, climate and fertilizer data"* (its own Kaggle
description) — it is generated, not measured from fields. Training and testing on one generated
table measures **internal consistency**, not whether a crop is actually grown there. Reporting
"99%" on it, then deploying that as advice, is the core overclaim the whole field is making.

**Problem B — no real soil, and no transfer test.** None of these projects use an actual soil
measurement (SoilGrids / spectra), and none tests whether a model trained in one region works in
another. A recommender that has never seen a held-out region has no evidence it generalizes.

**What TabPFN-3.5 changes — but not by itself.** Swapping the algorithm does *not* fix a synthetic
dataset. The real improvement is to fix the **data + validation**, and use TabPFN-3.5 where it is
genuinely strongest: small-n, mixed-type, missing-value tables, and — critically — a *new region
adapts from a handful of local samples* with no training.

## 3. What we build on top (improvement, not repetition)

| Axis | Existing projects | This project |
|---|---|---|
| Labels | one synthetic 2,200-row table | **real observed crop** (ESA WorldCereal: **6.83 M** points; US + Africa) |
| Soil features | none (only N/P/K columns) | **real soil** (ISRIC SoilGrids v2.0 API: pH, SOC, clay, N, CEC, BD) |
| Validation | random split on the same table | **cross-region transfer** (train US → test Africa and vice-versa) |
| Honesty | single headline accuracy | accuracy **beside** a majority baseline and a transfer gap |
| Adaptation | retrain the whole model | **few-shot**: a new region needs ~20 local points |
| Model | RF/SVM/XGB | **TabPFN-3.5** (explicit checkpoint), zero tuning |

## 4. Scientific incumbents we must respect (not just beat)

Real crop-suitability science already exists and is free:
- **FAO EcoCrop** — crop requirement database (temp, rainfall, pH, soil ranges per crop).
- **FAO/IIASA GAEZ** — Global Agro-Ecological Zones suitability maps.
- **CropSuite** (Scientific Reports 2020) — global crop-suitability maps.
- **iSDAsoil / Digital Earth Africa** — Africa soil + advisory layers.

None of these is a *small-n, soil-first, uncertainty-aware recommender a farmer can query by
coordinate*. Our lane is that intersection — and we cite these as the reference our numbers must
be consistent with, never claim to out-model them.

## 5. The one-line thesis

> Everyone reports ~99% on a synthetic crop table. Nobody shows a model built from **real soil and
> real crop labels**, and nobody reports **how badly it transfers** to a new region. TabPFN-3.5
> lets us do both cheaply — and the honest transfer number is the contribution.

Evidence files: `results/real_worldcereal_inventory.json`, `results/real_soilgrids_api_hitrate.json`,
`results/real_R2_*.json` (transfer), `docs/DATASETS.md`.
