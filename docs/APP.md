# The app and the agent

Two deliverables, one engine. Both call `soilhack.predict` (TabPFN-3.5, zero-shot) on **real**
data (ISRIC SoilGrids soil + NASA POWER climate) with the **real crop-label library** (ESA
WorldCereal).

- 🌐 **Live app:** https://soil2crop.nobar-party.cloud
- 🤖 **Agent:** `app/agent.py` (command line / importable)

See the top-level [`README.md`](../README.md) for setup and the deployment templates in
[`deploy/`](../deploy/).

---

## 1. Web app — Soil2Crop

**What it does:** type a farm coordinate → the app fetches real soil + real climate at that point,
fits TabPFN-3.5 on the *nearest real fields*, and returns a **ranked crop list** with calibrated
confidence, the **evidence** (the soil/climate values), and a **"how much to trust this"** panel.

**Honesty by design:**
- Every number is shown next to what carries it (context size, region mix, confidence gap).
- The confidence gap (top − 2nd) is displayed; a small gap means "ambiguous, don't trust blindly".
- A competence note states the measured caveat (zero-shot cross-continent transfer is weak;
  a few local points fix it).
- If a live API is slow, any missing feature is filled from the **nearest real field**; the UI
  badges each value `live` vs `nearest field` so it is never hidden.

**Architecture**

```
browser ──▶ Caddy (soil2crop.<your-domain>) ──▶ uvicorn :8877
                                                   │
            soilhack.predict ──┬─ SoilGrids REST  (real soil, live)
                               ├─ NASA POWER      (real climate, live)
                               ├─ reference_library.csv (2,181 real fields)
                               └─ TabPFN-3.5 resident model (warm once, ~1 s/query)
```

**Setup:** the simplest path is one command —

```bash
bash deploy/run_local.sh          # venv + deps + checkpoint + serve on :8877
```

or deploy as a service with `deploy/deploy.sh` + the Caddy vhost (see the README's
"Deploy it yourself"). The app resolves **all** paths relative to the repo or from env vars
(`SOILHACK_CKPT`, `SOILHACK_LIBRARY`, `SOILHACK_SRC`), so it runs from any directory.

---

## 2. Agent — predict → forecast → act

`app/agent.py` exposes three deterministic **tools** and an agent loop that **plans → calls → narrates**:

| Tool | Returns | Source |
|---|---|---|
| `predict(lat, lon)` | ranked crops + confidence + evidence | engine |
| `forecast(lat, lon)` | accuracy trajectory vs number of local rows | measured few-shot curve |
| `act(lat, lon)` | a dated, concrete next-action plan | engine + crop preferences |

**Design rule (the honesty guarantee): the language model may PHRASE; it may never INVENT a
number.** All facts are the tools' JSON. If an LLM endpoint is set
(`SOILHACK_LLM_BASE`/`_KEY`/`_MODEL`) it writes the closing narrative; otherwise a deterministic
template narrates. Either way the numbers are computed, never generated.

```bash
python app/agent.py "What should I grow at 9.40,-0.84 and what do I do next?"
```

Output includes a `trace` (which tools ran and whether they succeeded), the raw `facts`, and the
`answer` — so a reader can audit exactly which number came from where.

---

## 3. Map, suitability and transparency

- **Map** — a Leaflet map shows exactly where the queried coordinate is (OpenStreetMap tiles); it
  re-centres on the preset buttons and on lat/lon edits.
- **Suitability mode (all crops)** — pick any of **23 mainstream crops** in 8 groups (Cereal, Root,
  Legume, Cash, Perennial, Vegetable). Two clearly-labelled layers:
  - *Knowledge layer* (all crops): published **FAO EcoCrop** parameter ranges + Liebig's law of the
    minimum → Suitable / Marginal / Not suitable, with the **limiting factor** named
    (e.g. "annual rainfall 513 mm vs optimal 1200–2000 mm"). A **transparent rule, not a trained
    model**, and it carries **no accuracy claim**.
  - *Validated-ML cross-check* (only the 3 real-label crops): TabPFN-3.5's prediction is shown so the
    user sees when the learned model and the knowledge rule **disagree** (a transparency win).
  - If the chosen crop is unsuitable, the app suggests **better-suited alternatives** (same group
    preferred).
- **Transparency tab** — the app shows the **exact 7-step process** behind every number (input →
  fetch real features → build context → TabPFN-3.5 predict → calibrate → honesty layer →
  suitability) plus the measured performance, so a reviewer sees the mechanism, not just the result.

**Why this is not an overclaim:** adding 23 crops does **not** change the TabPFN-3.5 processing — the
ML layer still runs zero-shot on the **real 3-crop labels**; the 23-crop suitability is a separate,
explicitly **knowledge-based** rule with its own label. The two are visually separated so no reader
can mistake a rule for a validated prediction.

---

## 4. Security

- Strict input validation (lat/lon range, `k` clamped, crop name length).
- A small **per-IP rate limit** on `/recommend` and `/suitability`.
- Response hardening: `Content-Security-Policy`, `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy`.
- Errors return only an exception **type**, never a filesystem path or stack.
- The systemd unit runs with `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=full`, `ProtectHome`.
- **No secrets are in this repository.** Only `deploy/*.example` files ship.

---

## 5. Why this is "accurate, not just optimistic"

- **Real labels, real soil, real climate** — no synthetic Kaggle table.
- **Reported beside baselines** and with the **measured transfer gap** (`../RESULTS.md`).
- **The app tells the user when NOT to trust it** (competence + confidence gap), which is the part
  the ~99%-on-synthetic projects omit.
- **~5 local labelled points** (measured) turn a weak cross-continent prior into a solved
  recommendation — the farmer's own data is the value, cheaply.
