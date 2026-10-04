#!/usr/bin/env python3
"""Soil2Crop agent — predicts, forecasts and ACTS, with every number computed by the engine.

Design rule (honesty): the language model may PHRASE; it may never INVENT a number. All facts
come from three deterministic tools that wrap TabPFN-3.5 + the real-data engine:

  predict(lat, lon)          -> ranked crops + confidence            (the prediction)
  forecast(lat, lon)         -> how confidence grows with k local samples (the forecast)
  act(lat, lon)              -> a concrete, dated next-action plan    (the action)

If an LLM endpoint is configured (SOILHACK_LLM_BASE/KEY/MODEL), it writes the closing narrative
using ONLY the tool JSON as facts; otherwise a deterministic template narrates. Either way the
numbers are the engine's.

    python3 agent.py "what should I grow at 9.40,-0.84 and what do I do next?"
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
if os.environ.get("SOILHACK_SRC"):
    sys.path.insert(0, os.environ["SOILHACK_SRC"])

from soilhack import predict as P  # noqa: E402

# measured few-shot curve (RESULTS.md / real_R3b_summary.json): accuracy at k local rows
FEWSHOT_CURVE = {0: 0.460, 5: 1.000, 10: 1.000, 20: 1.000, 50: 1.000, 100: 1.000}


# ----------------------------------------------------------------- tools
def tool_predict(lat, lon):
    r = P.recommend(lat, lon)
    r.pop("explain", None)
    return r


def tool_forecast(lat, lon):
    """Honest forecast: the confidence/skill trajectory as local labelled rows are added."""
    base = P.competence(lat, lon)
    return {
        "lat": lat, "lon": lon,
        "curve_accuracy_vs_k_local_rows": FEWSHOT_CURVE,
        "reading": ("zero-shot (k=0) is at the majority baseline for a new continent; "
                    "≈5 local labelled points make the 3-class recommendation essentially solved"),
        "context_regions": base["nearest_context_regions"],
    }


def tool_act(lat, lon):
    """A concrete next-action plan, grounded in the prediction + the measured competence."""
    r = P.recommend(lat, lon)
    top = r["recommendations"][0]
    comp = P.competence(lat, lon)
    steps = [
        f"1. Candidate crop for {lat:.3f},{lon:.3f}: **{top['crop']}** "
        f"(confidence {top['confidence']*100:.0f}%).",
        "2. Before planting, verify with ~5-20 soil samples from THIS field "
        "(a phone photo of a soil-test sheet or a cheap NIR scan is enough) and re-run — "
        "the model re-fits in seconds with no retraining.",
        f"3. Soil check vs the crop's preference: soil pH/SOC/clay are {r['evidence']['soil']}; "
        "match to the crop's known range and correct with lime/gypsum/organic matter as needed "
        "(TabPFN predicts SUITABILITY, not a fertilizer dose — get an agronomist's dose).",
        "4. Re-scan one field per season to grow your local context; confidence rises with data.",
    ]
    return {"lat": lat, "lon": lon, "recommended_crop": top["crop"],
            "confidence": top["confidence"], "plan": steps,
            "honesty": comp["note"], "model": r["model"]}


TOOLS = {"predict": tool_predict, "forecast": tool_forecast, "act": tool_act}


# ----------------------------------------------------------------- LLM narration (optional)
def _llm_narrate(question, facts):
    base = os.environ.get("SOILHACK_LLM_BASE")
    key = os.environ.get("SOILHACK_LLM_KEY")
    model = os.environ.get("SOILHACK_LLM_MODEL", "gpt-4o-mini")
    if not base:
        return None
    try:
        import urllib.request
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content":
                 "You are Soil2Crop's assistant. You may ONLY use the FACTS JSON provided. "
                 "Never invent or round a number that is not in FACTS. Be concise and clear."},
                {"role": "user", "content": f"Question: {question}\n\nFACTS: {json.dumps(facts)}"},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return f"(LLM narration unavailable: {type(e).__name__})"


# ----------------------------------------------------------------- agent loop
def _coords(question):
    m = re.findall(r"-?\d+\.\d+", question)
    if len(m) >= 2:
        return float(m[0]), float(m[1])
    return None


def run(question, lat=None, lon=None):
    """Deterministic agent: plan tool calls, execute (each returns JSON facts), narrate."""
    if lat is None or lon is None:
        c = _coords(question)
        if not c:
            return {"error": "no coordinates found; pass lat/lon or include two decimals in the text"}
        lat, lon = c
    trace = []
    facts = {}
    for name in ("predict", "forecast", "act"):
        try:
            facts[name] = TOOLS[name](lat, lon)
            trace.append({"tool": name, "ok": True})
        except Exception as e:  # noqa: BLE001
            facts[name] = {"error": repr(e)}
            trace.append({"tool": name, "ok": False, "error": repr(e)})
    narration = _llm_narrate(question, facts) or (
        f"Top crop: {facts['predict'].get('recommendations',[{}])[0].get('crop','?')} "
        f"({facts['predict'].get('recommendations',[{}])[0].get('confidence',0)*100:.0f}%). "
        + (facts["act"].get("plan", ["—"])[0] if facts["act"].get("plan") else "—"))
    return {"question": question, "lat": lat, "lon": lon, "trace": trace,
            "facts": facts, "answer": narration,
            "disclaimer": "Numbers come only from the engine (TabPFN-3.5 + real data). Not agronomic advice."}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What should I grow at 9.40,-0.84 and what do I do next?"
    print(json.dumps(run(q), indent=1)[:4000])
