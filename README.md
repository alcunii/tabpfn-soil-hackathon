# 🌱 Soil2Crop — a real-data, transfer-honest crop recommender (TabPFN-3.5)

**Which crop grows here — built on real soil and real crop labels, with the accuracy *and the limits* measured, not promised.**

Most "crop recommender" projects report ~99% by training and testing on **the same synthetic 2,200-row Kaggle table**. That number measures internal consistency, not whether a crop actually grows there. Soil2Crop does the opposite: it uses **real** crop labels (ESA WorldCereal, 6.8 M points) and **real** soil (ISRIC SoilGrids) + **real** climate (NASA POWER), predicts with **TabPFN-3.5 zero-shot (no training)**, and then reports **how badly it transfers to a new region** — the number the field never publishes.

> ▶️ **Live demo:** https://soil2crop.nobar-party.cloud — type any coordinate.
> 🧠 **Model:** TabPFN-3.5 (`priorlabs/tabpfn` 9.0.0, checkpoint `tabpfn-v3.5-20260909.safetensors`), **zero-shot**.

---

## What it does

Enter a farm **coordinate** → Soil2Crop:

1. fetches **real soil** (pH, SOC, clay, N, CEC, bulk density) from the **ISRIC SoilGrids v2.0** API at that point;
2. fetches **real climate** (temperature, rainfall, seasonality, growing-degree-days, frost, aridity) from **NASA POWER**;
3. asks **TabPFN-3.5** (fit once on a library of **2,181 real fields**, no training) which crop the **nearest real fields** actually grow;
4. returns a **ranked crop list** with calibrated confidence, the **evidence** (the actual soil/climate values), and a **"how much to trust this"** panel;
5. offers a **suitability check for 23 mainstream crops** (FAO EcoCrop rules — a transparent *knowledge* layer, clearly separated from the trained model).

There are three tabs in the app:

- **① Which crop grows here?** → validated ML (TabPFN-3.5) on real labels.
- **② Can I grow a specific crop?** → knowledge-based suitability + alternatives.
- **③ How it works** → the exact process behind every number (transparency, so a judge can see the mechanism).

---

## The honest findings (why this is more than another 99%)

Every number below is a saved JSON in [`results/`](results/) — nothing is hand-typed. Full detail in [`RESULTS.md`](RESULTS.md).

- **The transfer gap (the finding).** A soil-only model looks strong on a random split — **0.918** accuracy vs a **0.385** majority baseline. But it **does not transfer across continents**: US→Africa is **0.490 vs 0.510** baseline (below chance), Africa→US **0.348 vs 0.345**. The ~99% synthetic numbers are an illusion of a self-referential benchmark.
- **Climate closes part of the gap — and a few local points close it fully.** Adding climate covariates lifts the random-split accuracy to **0.934**. A **few-shot** adaptation needs only **≈5 local labelled rows → 1.000** (measured **with spatial blocking**, so it is not leakage).
- **But we say what carries it.** Diagnostic: **climate alone** at k=5 already reaches 1.000, **soil alone** only 0.825 — the 3 classes are essentially *climate-zone proxies*. A plain **logistic regression at k=20 already hits 0.979**. So **TabPFN-3.5's real edge is needing ~5 rows with zero tuning, not a large accuracy margin.** We report it that way.
- **Regression (real spectra).** Predicting soil organic carbon from raw vis-NIR spectra: **R²=0.911, RPD=3.37 ("good")** at 280 context rows, vs a constant-baseline RMSE of 12.25 (3.4× better). The public pool is 64,211 spectra — the context is *data-limited, not model-limited*.

**Reading rules** (kept everywhere): never quote a skill number without its split/transfer and test set; every classification number sits beside its majority baseline; a number near its baseline is a **finding, not a failure**.

---

## Quick start (local, one command)

```bash
git clone https://github.com/alcunii/tabpfn-soil-hackathon
cd tabpfn-soil-hackathon
bash deploy/run_local.sh          # http://127.0.0.1:8877  (first request warms the model)
```

`run_local.sh` creates a venv, installs `requirements.txt`, downloads the TabPFN-3.5 checkpoint (~876 MB), and serves the app. **CPU is enough** — the context is narrow (13 features); the first request warms the model (~30–90 s), then queries are fast.

Prefer to run it as a service? See [**Deploy it yourself**](#deploy-it-yourself) below (systemd + Caddy on a **dedicated subdomain**).

### Run the notebooks (Colab-ready)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alcunii/tabpfn-soil-hackathon/blob/main/notebooks/01_feasibility_walkthrough.ipynb)

`notebooks/01_feasibility_walkthrough.ipynb` reproduces the headline experiments (TabPFN-3.5 fit → predict → metrics) on the committed samples; it runs on CPU.

---

## Deploy it yourself

The app is a plain FastAPI process; Caddy terminates TLS and reverse-proxies it. **No secrets exist in this repo** — the deploy files are generic templates.

```text
deploy/
├── Caddyfile.example       # dedicated-subdomain vhost + security headers
├── soil2crop.service       # systemd unit (uvicorn, hardened)
├── run_local.sh            # one-command local run
├── deploy.sh               # copy to /opt/soil2crop + install the unit
└── soil2crop.env.example   # env vars (checkpoint, port, optional agent LLM)
```

**1. Run the app** (on the host that will serve it):

```bash
DEST=/opt/soil2crop DOMAIN=soil2crop.example.com bash deploy/deploy.sh
```

**2. Point Caddy at it** (edit the domain first):

```bash
sudo cp deploy/Caddyfile.example /etc/caddy/soil2crop.caddy   # set your domain
sudo caddy validate --config /etc/caddy/soil2crop.caddy
sudo systemctl reload caddy
```

The example vhost already sets HSTS, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, strips `Server`, and only forwards the app's routes. Caddy issues the TLS certificate automatically.

### Configuration

| Env var | Meaning | Default |
|---|---|---|
| `SOILHACK_CKPT` | TabPFN-3.5 checkpoint path | `models/tabpfn-v3.5-20260909.safetensors` |
| `SOILHACK_LIBRARY` | reference-library CSV | `data/samples/reference_library.csv` |
| `TABPFN_MODEL_CACHE_SIZE` | resident-model cache | `2` |
| `PORT` | server port | `8877` |
| `SOILHACK_LLM_BASE/_KEY/_MODEL` | **optional** LLM to phrase the *agent* narrative (numbers are still engine-computed) | unset → deterministic template |

---

## Architecture

```text
        browser  ──▶  Caddy (soil2crop.<your-domain>)  ──▶  uvicorn :8877
                                                                │
                 soilhack.predict  ──┬─ ISRIC SoilGrids v2.0  (REAL soil, live)
                                     ├─ NASA POWER            (REAL climate, live)
                                     ├─ reference_library.csv (2,181 REAL fields)
                                     └─ TabPFN-3.5 resident model (fit once, ~1 s/query)
                                                                │
                 soilhack.crops    ────  FAO EcoCrop ranges  (KNOWLEDGE rule, not a model)
```

- **Zero-shot:** one `fit()` on the reference library, then a single forward pass per query. **No training loop anywhere.**
- **Resident-model pattern:** cost scales with *context size*, not query count — so many users share one fit.
- **Honest fallback:** if a live API is slow, a missing value is filled from the **nearest real field** and the UI badges it `nearest field` vs `live` — never hidden.

---

## The agent (predict → forecast → act)

[`app/agent.py`](app/agent.py) exposes three deterministic tools and an agent loop that **plans → calls → narrates**:

| Tool | Returns |
|---|---|
| `predict(lat, lon)` | ranked crops + confidence + evidence |
| `forecast(lat, lon)` | accuracy trajectory vs number of local rows (the measured few-shot curve) |
| `act(lat, lon)` | a dated, concrete next-action plan |

**Design rule:** the language model may *phrase*; it may **never invent a number**. All facts are the tools' JSON. An LLM endpoint is optional; without one, a deterministic template narrates.

```bash
python app/agent.py "What should I grow at 9.40,-0.84 and what do I do next?"
```

---

## The farmer's I/O contract (simple, free, global)

**The grower provides:** just a **latitude + longitude** (or "use my location"). *Nothing else required.*

**The grower gets:**
- a ranked crop list with calibrated confidences, e.g. `Maize 0.71 · Spring Cereals 0.18 · Winter Cereals 0.11`;
- **why** — the soil/climate values that drove it, in plain language;
- an **honesty line** — "this region is inside our tested area" or "no local validation here — treat as a prior";
- **one next action** — e.g. "collect ~20 local samples to sharpen this".

No account, no cost, works from a phone — because the heavy lifting is a **zero-shot TabPFN-3.5 call**, not a training pipeline the farmer runs.

---

## Repository layout

```text
README.md                 this file
RESULTS.md                measured results (every number from a JSON in results/)
LICENSE                   MIT
requirements.txt          TabPFN-3.5 + FastAPI + scientific stack

app/                      FastAPI web app + the predict→forecast→act agent
src/soilhack/             reusable library
  predict.py              real-data engine → TabPFN-3.5 → ranked crops
  crops.py                23-crop knowledge layer (FAO EcoCrop) — clearly not a model
  tabpfn_runner.py        explicit TabPFN-3.5 factory (checkpoint pinned)
  metrics.py              accuracy / macro-F1 / ECE; R² / RMSE / RPD / RPIQ
  data.py                 verified dataset loaders
scripts/                  experiment runners + figure builders
  run_feasibility.py      E1/E2 (crop rec; SOC from spectra)
  run_real_R2_fit.py      real soil→crop + cross-region transfer
  run_real_R3.py          climate + few-shot
  run_real_R3b_blocks.py  few-shot under spatial blocking (the honesty check)
  build_*.py / make_*.py  data builders + figures (figures read the JSONs only)
notebooks/                teaching notebooks (Colab-ready)
deploy/                   generic, secret-free deployment templates
data/samples/             small committed samples (real)
results/                  one JSON per experiment cell
figures/                  figures drawn from the result JSONs only
docs/                     DATASETS, IDEA, COMPETITIVE_ANALYSIS, APP
```

Large data is **gitignored** and re-downloadable from the public sources below.

---

## Datasets & provenance (all real, all open)

| Role | Source | Note |
|---|---|---|
| **Crop labels** | **ESA WorldCereal** (`data.source.coop/streambatch/worldcereal`) | 6.83 M real georeferenced points; 3 classes |
| **Soil** | **ISRIC SoilGrids v2.0** REST API | global 250 m; real values at any lat/lon |
| **Climate** | **NASA POWER** daily point API | free, keyless, 2019–2023 |
| *Reference (comparison)* | Kaggle `atharvaingle/crop-recommendation-dataset` | the **synthetic** 2,200-row table the field overuses |
| *Suitability rules* | **FAO EcoCrop** | crop requirement ranges (knowledge layer) |
| *Regression sample* | **OSSL** v1.2 vis-NIR | SOC from spectra |

Full inventory in [`docs/DATASETS.md`](docs/DATASETS.md).

---

## Security & privacy

- **No secrets in the repo.** Keys/paths live in a `.env` (gitignored) or a vault; only `*.example` files ship.
- **Dedicated subdomain** (`soil2crop.<your-domain>`) isolates the app; Caddy provides TLS + security headers.
- **App hardening:** strict input validation, a small **per-IP rate limit** on the compute endpoints, response hardening (CSP, `nosniff`, `DENY` framing, `no-referrer`), and the systemd unit runs with `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem/Home`.
- **Privacy:** the coordinate-only path needs **no account and stores nothing**; a farm's location stays in the request.
- **Errors** return only an exception *type*, never a filesystem path.

---

## How it maps to the hackathon criteria

| Criterion | This project |
|---|---|
| **Build an agent** | `app/agent.py` — predict → forecast → act. |
| **Build an extension or app** | The FastAPI web app + 3-tab UI + map, full setup below. |
| **Take on a hard problem** | Food security / soil health; real problems in the US *and* Africa. |
| **Formalize a new problem** | Real soil → real crop with an explicit **cross-region transfer protocol** (nobody else measures it). |
| **Showcase a harness** | TabPFN-3.5 behind a resident-model server, measured with baselines. |
| **Why TabPFN-3.5 specifically** | Small-n, mixed, missing-value tables; **zero tuning**; a new region adapts from ~5 local rows. |

---

## Limitations (stated, not hidden)

- WorldCereal gives **3 coarse classes**, not 22 species — we **scope to 3** and say so.
- The 23-crop suitability is a **knowledge rule**, not a trained model (labelled everywhere in the UI).
- SoilGrids is coarse (250 m) and its REST API is intermittent — handled with retries + honest fallback.
- Zero-shot **does not transfer across continents**; we report that as the headline finding.

---

## License & attribution

MIT (see [`LICENSE`](LICENSE)). Built with **TabPFN-3.5** by [Prior Labs](https://priorlabs.ai). Data © their respective providers (ESA WorldCereal, ISRIC, NASA, FAO). This project reports measurements on public data; it does not redistribute them.
