"""Dataset loaders — VERIFIED sources only.

Every loader either reads a local file (already staged) or downloads from a public,
license-checked URL. Large data is never committed; loaders that download go to data/raw/.
"""
import io
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
SAMPLES = os.path.join(ROOT, "data", "samples")
os.makedirs(RAW, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (soilhack)"}

# --------------------------------------------------------------- crop recommendation
CROP_REC_URLS = [
    "https://raw.githubusercontent.com/rhsbd/Crop-Recommendation-Using-Machine-Learning/main/Crop_recommendation.csv",
    "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv",
]
CROP_REC_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_crop_recommendation(path=None):
    """Kaggle `atharvaingle/crop-recommendation-dataset` (Apache-2.0, 2200 rows, 22 crops).

    Columns: N,P,K,temperature,humidity,ph,rainfall,label.
    Returns X (DataFrame), y (np.array of int class codes), and the class-name list.
    """
    if path is None:
        path = os.path.join(RAW, "crop_recommendation.csv")
    if not os.path.exists(path):
        last = None
        for url in CROP_REC_URLS:
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                with open(path, "wb") as f:
                    f.write(data)
                break
            except Exception as e:  # pragma: no cover
                last = e
        else:
            raise RuntimeError(f"could not download crop recommendation: {last}")
    df = pd.read_csv(path)
    classes = sorted(df["label"].unique())
    codes = {c: i for i, c in enumerate(classes)}
    X = df[CROP_REC_FEATURES].astype("float32").reset_index(drop=True)
    y = df["label"].map(codes).to_numpy()
    return X, y, classes


# --------------------------------------------------------------- OSSL spectra sample
OSSL_SAMPLE = os.path.join(SAMPLES, "ossl_validation_sample.csv")


def load_ossl_soc_sample(path=None):
    """400 real vis-NIR spectra, SNV-normalised, 1051 bands (400-2500 nm) + SOC label.

    Built from OSSL v1.2 (open). Columns: layer_id, dataset, lat, lon, country, SOC,
    log10_SOC, then `snv_<nm>nm` x 1051. Bands must be cut at 400 nm, never 350.
    Returns X (float32 ndarray), y (SOC), meta (DataFrame).
    """
    if path is None:
        path = OSSL_SAMPLE
    df = pd.read_csv(path)
    band_cols = [c for c in df.columns if c.startswith("snv_")]
    X = df[band_cols].to_numpy("float32")
    meta_cols = [c for c in ["layer_id", "dataset", "lat", "lon", "country", "SOC", "log10_SOC"] if c in df.columns]
    meta = df[meta_cols].copy()
    y = df["SOC"].to_numpy("float32")
    return X, y, meta


# --------------------------------------------------------------- generic csv helper
def read_csv_any(path, **kw):
    with open(path, "rb") as f:
        return pd.read_csv(io.BytesIO(f.read()), **kw)


def train_test_split_idx(n, frac=0.7, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    cut = int(round(frac * n))
    return idx[:cut], idx[cut:]
