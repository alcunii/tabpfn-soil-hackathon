#!/usr/bin/env python3
"""Soil2Crop web app — FastAPI server wrapping soilhack.predict.

Runs anywhere (Linux/macOS/Windows). Paths are resolved relative to this file
or via env vars, so nothing is machine-specific.

    python -m uvicorn app.server:app --host 127.0.0.1 --port 8877

Env:
    SOILHACK_CKPT     path to the TabPFN-3.5 checkpoint (default: ../models/...)
    SOILHACK_LIBRARY  path to the reference library CSV
    SOILHACK_SRC      optional explicit path to src/ (auto-detected otherwise)
    PORT              server port (default 8877)

Security: response security headers, per-IP rate limiting, strict input
validation. No secrets, no filesystem paths leaked in errors.
"""
import os
import sys
import time
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for cand in (os.environ.get("SOILHACK_SRC"), os.path.join(REPO, "src")):
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from soilhack import predict as P  # noqa: E402

app = FastAPI(title="Soil2Crop", version="1.0")
STATE = {"warmed": False, "error": None}

# --- simple in-memory per-IP rate limit (no external dependency) ------------
_WINDOW_S = 60
_MAX_REQ = 40
_HITS = defaultdict(deque)


def _rate_ok(ip: str) -> bool:
    now = time.time()
    dq = _HITS[ip]
    while dq and now - dq[0] > _WINDOW_S:
        dq.popleft()
    if len(dq) >= _MAX_REQ:
        return False
    dq.append(now)
    return True


@app.middleware("http")
async def _guard(request: Request, call_next):
    ip = request.client.host if request.client else "?"
    # only throttle the compute endpoints (not the static UI)
    if request.url.path in ("/recommend", "/suitability"):
        if not _rate_ok(ip):
            return JSONResponse({"error": "rate limit exceeded, slow down"}, status_code=429)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: https://*.tile.openstreetmap.org "
        "https://unpkg.com; style-src 'self' 'unsafe-inline' https://unpkg.com; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com; connect-src 'self'")
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return resp


@app.on_event("startup")
def _warm():
    try:
        info = P.warm()
        STATE["warmed"] = True
        STATE["info"] = info
        print("warmed:", info, flush=True)
    except Exception as e:  # noqa: BLE001
        STATE["error"] = type(e).__name__
        print("warm failed:", type(e).__name__, e, flush=True)


class Query(BaseModel):
    lat: float
    lon: float
    k: int = 300


class SuitQuery(BaseModel):
    lat: float
    lon: float
    crop: str | None = None


def _in_range(lat, lon):
    return -90 <= lat <= 90 and -180 <= lon <= 180


@app.get("/health")
def health():
    return {"status": "ok", "warmed": STATE["warmed"], "info": STATE.get("info"),
            "error": STATE.get("error"), "model": "TabPFN-3.5 (zero-shot)"}


@app.post("/recommend")
def recommend(q: Query):
    if not _in_range(q.lat, q.lon):
        return JSONResponse({"error": "lat/lon out of range"}, status_code=400)
    k = max(20, min(int(q.k), 1200))
    try:
        out = P.recommend(q.lat, q.lon, k=k)
        out["competence"] = P.competence(q.lat, q.lon)
        return out
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": type(e).__name__}, status_code=500)


@app.get("/method")
def method():
    """Transparency: the exact process behind every number (shown in the app)."""
    return {
        "engine": "TabPFN-3.5 (Prior Labs), zero-shot classifier — no training loops anywhere.",
        "data": {
            "labels": "ESA WorldCereal — 6.83M real crop points (3 classes: Maize, Winter Cereals, "
                      "Spring Cereals); reference library = 2,181 real fields",
            "soil": "ISRIC SoilGrids v2.0 REST (pH, SOC, clay, nitrogen, CEC, bulk density; 0-5 cm) — live",
            "climate": "NASA POWER daily point API 2019-2023 (temp, rainfall, GDD, frost, aridity) — live",
        },
        "pipeline": [
            "1. INPUT — the user gives a farm coordinate (lat, lon).",
            "2. FETCH REAL FEATURES — soil is pulled live from SoilGrids, climate from NASA POWER. "
            "If a live call is slow, a missing value is filled from the nearest real field and marked "
            "'nearest field' (never hidden).",
            "3. BUILD CONTEXT — the ~300 nearest real fields (real labels + real soil + real climate) "
            "become the TabPFN context. TabPFN cost scales with context size, not query count.",
            "4. TABPFN-3.5 PREDICT — the model is fit ONCE on the reference library (resident model) and "
            "predicts the query in a single forward pass. Zero-shot: no gradient training, no tuning.",
            "5. CALIBRATE — softmax probabilities are the confidences; the app shows the gap between "
            "the top two so a narrow gap reads as 'ambiguous'.",
            "6. HONESTY LAYER — a competence note from the measured experiments: zero-shot transfer "
            "across continents is ≈ the majority baseline; ~5 local observations make the 3-class "
            "recommendation essentially solved (carried mostly by climate).",
            "7. SUITABILITY (all crops) — a separate KNOWLEDGE rule (FAO EcoCrop ranges, Liebig "
            "minimum) scores any of 23 crops for the point; it is NOT a trained model and carries no "
            "accuracy claim. For the 3 real crops it is cross-checked against the TabPFN prediction.",
        ],
        "measured": {
            "3_crop_random_split_accuracy": 0.918, "majority_baseline": 0.385,
            "us_to_africa_transfer": 0.490, "few_shot_k5_accuracy": 1.000,
            "note": "See RESULTS.md; every number is a saved JSON in the repo.",
        },
    }


@app.get("/crops")
def crops_list():
    from soilhack import crops as C
    return {"groups": C.crop_groups(), "crops": sorted(C.REQUIREMENTS), "source": C.SOURCE}


@app.post("/suitability")
def suit(q: SuitQuery):
    if not _in_range(q.lat, q.lon):
        return JSONResponse({"error": "lat/lon out of range"}, status_code=400)
    crop = q.crop[:40] if q.crop else None
    try:
        return P.suitability(q.lat, q.lon, crop=crop)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": type(e).__name__}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/results", response_class=HTMLResponse)
def results():
    """The measured-results landing page (single-file HTML, figures inlined)."""
    path = os.path.join(HERE, "results.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Soil2Crop results</h1><p>See RESULTS.md in the repository.</p>")
