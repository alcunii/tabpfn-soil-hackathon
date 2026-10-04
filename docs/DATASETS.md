# Dataset inventory for soil / agriculture (farmer-facing)

Availability was **verified** from the GPU box's own internet link (GitHub/HF/UCI/PyPI
reachable; Kaggle/Zenodo need auth or a GET). "Verified" below means we actually reached the
resource (for the two we ran: the crop-recommendation CSV and the OSSL sample are **staged and
used**). Anything marked *unverified* still needs a download test before it can back a claim.

Legend: **✅ verified reachable/used** · **⚠ needs API key / registration** · **❔ unverified**

---

## A. Datasets we have ALREADY RUN (evidence: `results/`)

| Dataset | Provider | Rows × cols | Task | License | Status |
|---|---|---|---|---|---|
| Crop Recommendation | Kaggle `atharvaingle/crop-recommendation-dataset` (mirror: GitHub `rhsbd`) | 2,200 × 7 | 22-crop classification | Apache-2.0 | ✅ used in E1 |
| OSSL v1.2 vis-NIR (+MIR, soil lab/site) | Open Soil Spectroscopy Library | 135,651 layers; 64,211 vis-NIR+SOC | SOC/pH/clay regression from spectra | open | ✅ sample used in E2; full pool local on VPS |

The **full OSSL** is on the VPS at `the OSSL v1.2 release` (vis-NIR 182 MB gz, MIR 420 MB gz,
soil lab, soil site) and inventoried at `/root/ossl_inventory/`. This is the strongest asset: it
lets a *farmer-facing* spectral model be built with a 64k-row open reference library.

## B. Strong candidates for the remaining ideas (need one download test)

| Dataset | Provider | Size | Task it enables | License | Status |
|---|---|---|---|---|---|
| Crop Health & Environmental Stress | Kaggle `datasetengineer` | 212,019 × 32 | crop-health class + yield; soil+weather+RS | CC BY-NC-SA 4.0 | ❔ (doc fetched) |
| Soil Nutrient Gap Prediction (Sustainable Maize) | Kaggle competition | — | nutrient-gap regression + fertilizer rec | competition rules | ❔ |
| Agricultural Yield Prediction | Kaggle `zoya77` | small | yield regression from soil+climate | see page | ❔ |
| Smart Farming Sensor Data for Yield | Kaggle `atharvasoundankar` | 28 kB | sensor → yield | see page | ❔ |
| Crop Yield Prediction (soil+weather) | Kaggle `gurudathg` | small | yield regression | see page | ❔ |
| Indian Rice Yield & Climate 2000–2015 | Kaggle `prince1125` | 585 kB | yield vs climate | see page | ❔ |

Kaggle needs credentials (`~/.kaggle/kaggle.json` or `KAGGLE_KEY` env). **Not present on the
workstation or the VPS** (verified). Two options: (a) the user adds a Kaggle API token, or (b) we
prefer GitHub/HF/UCI mirrors of the same data (already the case for crop recommendation).

## C. Open geospatial / provider references (big, authoritative, not "a Kaggle table")

| Provider | What | Note |
|---|---|---|
| **SoilGrids / ISRIC WoSIS** | global 250 m maps of 14 properties + ~80k profile points | WoSIS point data is the open *training* set; REST API currently paused |
| **iSDAsoil** (AWS Open Data, `s3://isdasoil`) | 30 m soil property maps for Africa; ~100k–130k training samples; free API | built for smallholder advisory — the exact mission of this project |
| **LUCAS Soil** (ESDAC) | EU topsoil 2009/2015/2018, ~19k–40k samples, OC/pH/NPK/texture | registration required; **already inside OSSL** (LUCAS = vis-NIR only there) |
| **OpenLandMap** | soil organic carbon at depths, 250 m | GEE/Cloud-Optimized |
| **OSSL (v1.2/v1.3)** | global harmonized spectra + labels | our primary asset |
| **AfSIS / iSDA profile data** | Africa soil profiles | GitHub + OSF DOI 10.17605/OSF.IO/A69R5 |

## D. Why these fit TabPFN-3.5 specifically

The farmer reality is **small n per farm, wide correlated features, mixed types, missing values,
no tuning budget.** That is precisely where TabPFN-3.5 is strongest and where gradient boosting
needs the most hand-holding. The datasets above were chosen because they let us *demonstrate* that
regime (a tiny context still predicts well) rather than hide from it behind a large clean table.

## E. Data rule (binding)

Large raw data is **gitignored**. The repo commits only: small samples (`data/samples/`), the
result JSONs (`results/`), and the loaders that can re-download from the verified URLs.
