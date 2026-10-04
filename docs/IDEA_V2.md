# Idea v2 — "Soil2Crop": a real-data, transfer-honest crop recommender for the US and Africa

This supersedes the menu in `IDEAS.md` §6. It is the synthesis the user asked for: combine
**OSSL soil prediction** with **crop recommendation**, start from **real problems**, **improve on
existing projects** (not repeat them), address **dataset limitations honestly**, and define a
**simple, cheap, global** input→output.

Read with: `docs/COMPETITIVE_ANALYSIS.md` (the gap, verified), `docs/DATASETS.md`, `RESULTS.md`.

---

## 1. Start from the real problem (why this matters)

**Problem 1 — the "99% crop recommender" doesn't work on a real farm.** Every popular project
(Harvestify, dozens of papers) scores ~99% on one **synthetic** 2,200-row table (`COMPETITIVE_ANALYSIS.md`
§1). It never used real soil and never tested a new region. A grower in Ghana or Iowa who follows it
is being advised by a model whose accuracy is an artifact of how the table was generated.

**Problem 2 — the data the recommender needs, a smallholder cannot afford.** A crop recommendation
needs soil (N/P/K/pH/OC) + climate. A lab soil test is ~$20-50/sample and repeated across a field.
Smallholders in **Africa** (and many US small farms) simply don't have it, so they over- or
under-apply fertilizer — lost yield, lost money, degraded soil.

**Problem 3 — advice doesn't transfer across regions.** Africa and the US have completely different
soils, climates and cropping systems. A model that works in Iowa has *no evidence* it works in
Ethiopia. Nobody measures that transfer gap, so nobody knows how much to trust cross-region advice.

**Our real problem to solve:** give a grower *anywhere* an **honest, soil-first crop recommendation**
whose accuracy is reported **on real data**, **next to a baseline**, and **with a measured transfer
gap** — so the tool can say "trust me here, don't here."

## 2. The idea

**Soil2Crop** predicts soil properties from whatever soil info is available, then recommends which
crop is actually grown/suitable at that location — trained and validated on **real** data.

Two soil input paths (cheap-first), one output:

| Soil input | Cost to grower | What we do |
|---|---|---|
| **Just a coordinate** | **free** | pull real soil from **ISRIC SoilGrids v2.0** (pH, SOC, clay, N, CEC, BD) |
| **A cheap NIR scan** | ~device cost | **TabPFN-3.5 spectral model** predicts SOC/pH/clay (proven: R²=0.91, RPD 3.37) |
| *(optional) a lab test* | $20-50 | paste N/P/K/pH directly |

Output: a **ranked crop list** for that point, each with a **calibrated confidence**, plus the
**evidence** (which soil variables drove it) — and a **honesty flag**: whether this region is inside
the model's *measured* competence (from the transfer test), or outside it ("we have no local
validation here — treat as a prior, get 20 local samples to sharpen").

## 3. How it improves on the incumbents (not a re-run)

| Axis | Harvestify / papers | Soil2Crop |
|---|---|---|
| Labels | synthetic 2,200-row table | **real observed crops** — ESA WorldCereal (6.83 M points, US + Africa) + CropSight-US / USDA CDL |
| Soil | none (N/P/K columns only) | **real** SoilGrids v2.0 API + **OSSL spectra** model |
| Validation | random split, same table | **cross-region transfer** US↔Africa, **+ majority baseline** |
| Honesty | one 99% headline | accuracy **beside** baseline, **transfer gap**, competence flag |
| Adaptation | retrain everything | **few-shot**: ~20 local rows re-fit TabPFN, no training |
| Model | RF/SVM/XGB (tuned) | **TabPFN-3.5** (explicit checkpoint), zero tuning |
| Reach | one region | **global** (query any coordinate), tested on US + Africa |

## 4. Address every dataset limitation (explicitly)

| Limitation | Reality | Our handling |
|---|---|---|
| **WorldCereal is 3 coarse classes** (Maize / Winter Cereals / Spring Cereals) | it is a global crop map, not 22 species | we **scope to 3 crops** and say so; we do NOT claim 22-crop skill on real data |
| **The 22-crop Kaggle set is synthetic** | good for a *self-consistency* demo, not for advice | use it only as a **baseline comparison** to *quantify* how over-optimistic synthetic benchmarks are |
| **SoilGrids REST has ~20% null/miss + 500/429 on heavy queries** | measured 13/16 hit-rate; flaky | query **one small property set**, retry, log coverage **per experiment**; fall back to iSDA/OSSL tile |
| **Soil is coarse (250 m)** | a 250 m soil value is a field-scale prior, not per-plant truth | state it; pair with a local scan/test when the grower has one |
| **Soil alone cannot fully determine crop** (climate, market, land use matter) | soil is necessary, not sufficient | frame output as *soil-suitability* + confidence, never "you must grow X"; add climate covariates in the build phase |
| **Coordinates are sensitive** | a farm's location is private | the app keeps coordinates local; no upload required for the coordinate-only path |
| **No Africa OSSL coverage at field scale** | OSSL spectra are sparse in Africa | the few-shot path is exactly the fix: 20 local samples adapt the model |

## 5. Simple, cheap, global — input & output (the product contract)

**The grower provides (simplest possible):**
- **Path A (free):** latitude + longitude (or "use my location"). *Nothing else required.*
- **Path B (cheap):** the same point + a photo of a $cheap NIR scanner readout, or a lab test if they have one.

**The grower gets:**
- a ranked **3-crop list** with confidences, e.g. *Maize 0.71 · Spring Cereals 0.18 · Winter Cereals 0.11*;
- **why** (the soil values that drove it, in plain language: "low nitrogen, pH 5.1 → favors cereals");
- an **honesty line**: "this region is inside our tested area (accuracy 0.xx)" or "no local validation — treat as a prior";
- **one next action**: "get 20 local samples to sharpen this" / "the recommended N top-up is ~X kg/ha".

No account, no cost, works from a phone — because the heavy lifting is a **zero-shot TabPFN-3.5
call**, not a training pipeline the farmer would have to run.

## 6. What TabPFN-3.5 makes possible here (mechanism, not hype)

- **Zero-shot with tiny n:** a new region becomes usable from ~20-100 labelled points — no training,
  no GPUs, no MLOps. (Measured: crop-rec at 98.5% from 100 rows; SOC R²=0.91 from 100 spectra.)
- **Mixed, missing, messy inputs:** soil tables are full of missing values and mixed types — TabPFN's
  native regime (no imputation pipeline needed).
- **Calibrated confidence:** measured ECE ≤ 0.03 → the confidence we show the farmer is trustworthy
  (this is what lets the "competence flag" be honest).
- **Per-query cost is set by CONTEXT, not query count** → serving many farmers from one library is
  cheap; the reference library is the asset, not a trained model per user.

## 7. Architecture (maps to 3 hackathon tracks at once)

```
  grower: coordinate (± scan/test)
      │
      ├─► [soil layer]  SoilGrids v2.0 (free)  OR  OSSL TabPFN-3.5 spectral model
      │
      ├─► [crop layer]  TabPFN-3.5 classifier: soil -> real crop labels (WorldCereal)
      │        · few-shot: a handful of LOCAL labelled points re-fit the model for that region
      │
      ├─► [honesty layer] competence flag from the US↔Africa transfer experiment + ECE
      │
      ▼
  ranked crops + confidence + reasons + one next action
```

- **Formalize a new problem** — real soil→real crop with a transfer protocol (the framing).
- **Extension / app** — a coordinate→recommendation web app / MCP server.
- **Build an agent** — an agent that (predict → check competence → act: emit the next-action plan).

## 8. Honest expected results (to be measured in the build phase)

- The **real-data transfer gap** is likely *large* (US↔Africa), and that is the contribution — we
  report it, we do not hide it. `results/real_R2_*.json`.
- The **synthetic 99%** will beat the real-data number: we quantify the gap between the two and say
  plainly which one a farmer should trust.

## 9. Next steps (if approved)

1. Finish R2 (real soil→real crop transfer) + add a **majority baseline** and **within-region** cell.
2. Add **E5**: full-OSSL spectra → soil (upgrade E2's 400-row sample to the 64k pool).
3. Add **E6**: messy-data robustness (missing values, unit drift).
4. Build the app (coordinate → recommendation) + agent wrapper.
5. Publish results page + record the video.
