# The four idea candidates

Each idea is farmer-facing, TabPFN-native (small n, mixed types, missing values, zero tuning),
and mapped to one of the hackathon's tracks. The **evidence** column points to a real result file
in `results/` — read `RESULTS.md` for the numbers.

Legend for evidence confidence:
**● measured here** · **◐ plausible, needs one experiment** · **○ needs new data**

---

## Idea 1 — **SoilSense**: few-shot soil-property prediction from cheap spectra
**Track:** *Formalize a new problem* (take a domain dataset nobody treats as a clean tabular task
and turn it into a proper prediction task).

**The problem.** Smallholder growers in the US and globally cannot afford a $30-per-sample wet-lab
soil test repeated across a field. A handheld vis-NIR scanner is cheap; turning a scan into
*trustworthy* SOC/pH/clay numbers usually needs thousands of labelled samples and a trained model.
TabPFN flips that: a handful of local reference samples become the *context*, and the model predicts
the rest **zero-shot**.

**What we build.** An app/MCP server: scan → TabPFN-3.5 → soil property + a confidence. Ships with a
built-in open reference library (OSSL, 64k spectra) so a new user is usable **before** they own any
data, then sharpens as their own samples accumulate.

**Why TabPFN.** Spectra are 1,000+ wide, correlated columns; sample counts per farm are tiny;
there is no tuning budget. That is exactly TabPFN-3.5's regime.

**Evidence.** ● `feas_e2_*` — **SOC R² = 0.911, RPD = 3.37 ("good")** from 1,051 raw SNV bands at
**280 context rows**, vs constant-baseline R² ≈ 0. The full OSSL SOC pool is 64,211 spectra.

**Impact.** Cheaper, denser soil carbon mapping for growers; usable from day one.

---

## Idea 2 — **NutriRx**: soil test → fertilizer & lime prescription
**Track:** *Build an extension or app* (a visual app / MCP server from our repo).

**The problem.** A soil lab returns numbers (N, P, K, pH, OC…). The grower still has to translate
them into *"how much urea, how much lime, what to plant."* That translation is where money and
yield are won or lost, and it is exactly the step small farms lack an agronomist for.

**What we build.** A local web app + MCP server: paste a soil test (or scan), get (a) a **crop
recommendation**, (b) a **fertilizer prescription** toward target NPK, (c) a **lime/gypsum**
recommendation for pH, each with the TabPFN confidence and the *reasons* (which inputs drove it).

**Why TabPFN.** Recommendation + prescription are small-n tabular problems with mixed types and
missing values; the whole value is a *trustworthy, explainable* answer, not a leaderboard score.

**Evidence.** ● `feas_e1_*` — **22-crop recommendation at 99.4–99.6% accuracy** (chance 4.5%) from 7
soil/climate features, ECE ≤ 0.03. ◐ The lime/NPK *prescription* half needs one more experiment
(E3: predict lime requirement / nutrient gap) — dataset candidates in `docs/DATASETS.md`.

**Impact.** Directly the grower's decision: what to plant and what to spread. This is the most
demo-friendly idea and the easiest video.

---

## Idea 3 — **SoilAdvisor**: an agent that predicts, forecasts and *acts*
**Track:** *Build an agent* (put TabPFN-3.5 behind an agent that predicts, forecasts and acts).

**The problem.** A grower doesn't want a number, they want a *decision loop*: "my field reads like
this — what should I do this week, and what will it do to my yield and my soil carbon?"

**What we build.** An agent that: (1) **predicts** current soil properties from a scan/test,
(2) **forecasts** yield or SOC trajectory under a proposed practice change,
(3) **acts** — emits a concrete, dated plan (apply lime now, cover-crop in October, retest in spring)
and logs it. Every action is grounded in a TabPFN-3.5 prediction with an uncertainty band.

**Why TabPFN.** The predict step is ideas 1/2; the forecast step turns it into a sequence of small
tabular predictions; the "act" step is an LLM planner wrapped around deterministic predictions —
so the agent *shows its work* instead of hallucinating.

**Evidence.** ◐ Builds directly on ● E1 + ● E2 (both measured). ⚠ The forecast step needs an
external check: an LLM's "forecast" is only trustworthy if every number it prints comes from a
TabPFN call, never from the model itself. Design rule: **the agent may phrase, the model must
compute.**

**Impact.** Highest ceiling for "wow" (an agent that acts), highest risk of overclaiming. Best as a
*combined* demo on top of a chosen core idea.

---

## Idea 4 — **FieldAlert**: crop-stress / yield early warning
**Track:** *Take on a hard problem* (health, **climate**, or a problem you care about).

**The problem.** Yield losses are usually visible only after the stress has already cost the crop.
A grower wants an early warning: "this field is heading toward a stress/yield-loss zone."

**What we build.** A model over per-field tabular records (weather, soil moisture/pH/OC, NDVI,
canopy, pest indicators) that flags stress risk and forecasts yield, with a per-field confidence
and a "what changed" attribution.

**Why TabPFN.** Per-field records are small, heterogeneous, missing-prone — the regime TabPFN wins.
TabPFN-**TS** (the time-series sibling) is also relevant if we build the series version.

**Evidence.** ○ Needs a new dataset. Best candidate measured feasible:
`Crop Health and Environmental Stress Dataset` (212,019 rows, 32 cols, soil+weather+RS, binary
crop-health label) — see `docs/DATASETS.md`. One experiment (E4) would turn it into evidence.

**Impact.** Climate-resilience framing scores well with judges; data quality of synthetic datasets
is the risk to disclose honestly.

---

## §6 — Recommendation (my read, for you to overrule)

Two of these are strong on *both* axes (technical grounding + farmer impact):

- **SoilSense (1)** is the most *technically defensible* — we already have a "good" RPD result on
  real spectra and a clear, novel formulation. It fits the "formalize a new problem" track exactly.
- **NutriRx (2)** is the most *demo-friendly and farmer-direct* — we already have a 99% crop
  recommender, and the prescription half is one experiment away.

My suggestion: **build SoilSense as the core (it has the measured depth) + NutriRx as the
user-facing app layer (it has the demo)** — they share the same scan→prediction engine, so together
they hit two tracks at once. FieldAlert (4) is the best *stretch* third track if we want a climate
angle. SoilAdvisor (3) is the wrapper that could turn the whole thing into an "agent that acts."

Nothing is chosen yet — this is the menu.
