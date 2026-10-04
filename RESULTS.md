# Measured results — soilhack (TabPFN-3.5)

Every number is copied from a JSON in `results/`. None is estimated.
Model: **TabPFN-3.5** (`tabpfn 9.0.0`, checkpoint `tabpfn-v3.5-20260909.safetensors`),
device **cuda** (an NVIDIA GPU (17 GB VRAM)) on the GPU workstation.
One `fit()` per cell — **no training loop anywhere**.

---

## R2 — REAL soil -> REAL crop, and the transfer gap (the headline)

**Data:** real crop labels (**ESA WorldCereal**, satellite-derived) + real soil (**ISRIC SoilGrids
v2.0** API: pH, SOC, clay, N, CEC, BD at 0-5 cm). 2,163 rows with full soil (3.9% coverage loss
logged). 3 real crops: Maize, Winter Cereals, Spring Cereals. No Kaggle, no synthetic labels.

| Cell | Train | Test | Accuracy | Macro-F1 | Majority baseline | Skill? |
|---|---|---|---|---|---|---|
| `random_mixed` | US+Africa 70% | US+Africa 30% | **0.918** | 0.919 | 0.385 | YES strong |
| `withinUS` | US 70% | US 30% | 0.864 | 0.862 | 0.322 | YES strong |
| `withinAfrica` | Africa 70% | Africa 30% | 0.496 | 0.332 | **0.496** | NO no skill |
| `trainUS_testAfrica` | US (all) | Africa (all) | 0.490 | 0.329 | **0.510** | NO below baseline |
| `trainAfrica_testUS` | Africa (all) | US (all) | 0.348 | 0.178 | 0.345 | NO no skill |

**The finding that matters (a correction, not a triumph):**

1. On a **random split** a soil-only TabPFN-3.5 model looks **91.8%** (vs 38.5% baseline) — real
   soil->crop signal exists, and TabPFN-3.5 finds it with zero tuning.
2. But **within Africa there is no skill** (49.6% vs 49.6% baseline), and **across regions the model
   is at or below the majority baseline** (US->Africa 0.490 vs 0.510; Africa->US 0.348 vs 0.345).
3. Therefore: **a soil-only crop recommender does not transfer across continents, and on this task
   is not even better than the majority crop within Africa.** The random-split number is
   *optimistic* — exactly the illusion the synthetic 99% projects sell.

This is the project's honest contribution: **publish the transfer gap the field never measures.**
It also says what the build must add (`docs/IDEA_V2.md` §6): climate covariates, a few-shot local
context, and a competence flag that refuses to claim skill where it has none.

Reference point: the **synthetic** crop-recommendation table (2,200 rows, 22 crops) gives
**0.994** on a random split (E1) — with **no** transfer test at all.

---

## R3 / R3b — does climate close the gap, and how little local data is enough?

Added **climate covariates** (NASA POWER daily 2019-2023, 7 features: mean/range temp, annual
precip, precip seasonality, growing-degree-days, frost days, aridity — 100% coverage over a
0.25-deg grid). Then measured a **few-shot learning curve**: adapt to Africa using only *k* local
labelled rows. R3b repeats the few-shot under **spatial blocking** (train-local and test rows in
different 0.5-deg cells) to rule out location leakage.

| Cell | Accuracy | Macro-F1 | Majority baseline | Note |
|---|---|---|---|---|
| `soilonly_randomsplit` | 0.925 | 0.928 | 0.385 | soil only, random split |
| `soilclim_randomsplit` | **0.934** | 0.936 | 0.385 | + climate → small gain |
| `soilclim_US2AF` (zero-shot transfer) | 0.488 | 0.328 | 0.511 | still below baseline |
| `soilclim_AF2US` (zero-shot transfer) | 0.334 | 0.167 | 0.342 | still no skill |
| `fewshot_blocked_k0` (US only) | 0.460 | 0.315 | 0.540 | no local data → no transfer |
| **`fewshot_blocked_k5`** | **1.000** | 1.000 | 0.540 | **5 local rows → solved** |
| `fewshot_blocked_k10..100` | 1.000 | 1.000 | 0.540 | saturates immediately |
| `climateonly_k5` (diagnostic) | **1.000** | 1.000 | 0.540 | climate ALONE explains it |
| `soilonly_k5` (diagnostic) | 0.825 | 0.824 | 0.540 | soil alone is weaker |
| `baseline_lr_k20` (diagnostic) | 0.979 | — | 0.540 | logistic regression, 20 local rows |

**Leakage ruled out:** in the *unblocked* few-shot, **70.6%** (527/746) of test rows shared a
0.5-deg block with a local training row — a real leakage risk. Under **spatial blocking** the
result is **identical (1.000 at k=5)**, so the skill is genuine, not leakage.

**Why 1.000 is legitimate (and honestly bounded):** the diagnostic shows **climate alone at k=5
already reaches 1.000** while **soil alone is 0.825**. The 3 WorldCereal classes (Maize vs
Winter/Spring Cereals) are essentially **climate-zone proxies**, so with a handful of local
climate+label points the mapping is learned perfectly. This is **not** a claim that TabPFN is
magically superior — a plain **logistic regression at k=20 already hits 0.979**. TabPFN's real
edge is needing only **~5 rows and zero tuning**, not a large accuracy margin.

**Net:** zero-shot does not transfer across continents; **a few local labelled points (≈5) make the
recommendation essentially solved for these classes, carried mostly by climate.** That is the
"accurate one" — accurate *because* it is honest about needing local data and about what carries it.

---

## E1 — Crop recommendation from 7 soil+climate features (synthetic table, for comparison)

7 features -> 22 crops. 70/30, seed 42, 660 test rows. **Chance = 4.55%.**

| Context rows | Accuracy | Macro-F1 | Top-3 | Mean conf | ECE |
|---|---|---|---|---|---|
| 100 | 0.9848 | 0.984 | 1.000 | 0.958 | 0.028 |
| 500 | 0.9955 | 0.995 | 1.000 | 0.991 | 0.007 |
| 1,000 | 0.9955 | 0.995 | 1.000 | 0.994 | 0.004 |
| 1,540 (all) | 0.9939 | 0.994 | 1.000 | 0.995 | 0.007 |

**Read this as the *contrast*, not the headline:** ~99% on a *synthetic* table vs the *real* transfer
numbers above. The gap between them is the point.

## E2 — Soil organic carbon from vis-NIR spectra (regression)

1,051 SNV bands (400-2500 nm) -> SOC (%). 400-spectrum open OSSL sample, 70/30, seed 42, 120 test.
Constant (train-mean) baseline = collapse detector.

| Context rows | R2 | RMSE | RPD | RPD read | Bias | Constant RMSE |
|---|---|---|---|---|---|---|
| 100 | 0.903 | 3.82 | 3.22 | good | +0.22 | 12.58 |
| 280 (pool max) | **0.911** | 3.65 | **3.37** | **good** | -0.64 | 12.25 |

**Finding:** TabPFN-3.5 predicts SOC from raw spectra at **R2=0.91, RPD=3.37 ("good")** while the
constant baseline scores R2~0 (3.4x error reduction). The 400-row sample caps context at 280; the
full OSSL SOC pool is **64,211** spectra, so the context is *data-limited, not model-limited*.

## R1 — Real global soil features (source viability)

| Metric | Value |
|---|---|
| Source | ISRIC SoilGrids v2.0 REST API, depth 0-5 cm, mean |
| API hit-rate (random US + Africa points) | **13/16 non-null (81%)** |
| Soil coverage on the 2,250 WorldCereal points | **2,163 / 2,250 = 96.1%** |
| Heavy multi-property query (9 props) | intermittent — HTTP 500/429 (logged) |
| Reference | `results/real_soilgrids_api_hitrate.json`, `results/real_R2_features_summary.json` |

---

## Reading rules (do not break these)

1. Never quote a skill number without its **split/transfer** and **test set** next to it.
2. Every classification number is reported **beside its majority baseline**.
3. Report `random_mixed` and the **transfer** cells **together** — the gap is the finding.
4. A number near its baseline is a **finding, not a failure**.
5. A synthetic-table number is labelled as such and never presented as real-world skill.
