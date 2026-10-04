"""Metrics for the soilhack feasibility experiments.

Classification: accuracy, macro-F1, top-k, mean confidence, ECE.
Regression:     R2, RMSE, MAE, RPD, RPIQ (RPD/RPIQ interpretations per Chang et al. 2001,
                reported together because extractable nutrients are strongly right-skewed).
"""
import numpy as np


# ------------------------------------------------------------------ classification
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def top_k_accuracy(proba, y_true, k=3):
    proba = np.asarray(proba)
    classes = np.argsort(-proba, axis=1)[:, :k]
    y = np.asarray(y_true).reshape(-1, 1)
    return float(np.mean((classes == y).any(axis=1)))


def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def expected_calibration_error(proba, y_true, n_bins=15):
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc_b = correct[m].mean()
        conf_b = conf[m].mean()
        ece += (m.sum() / n) * abs(acc_b - conf_b)
    return float(ece)


def classification_report(proba, y_true, n_classes=None, extra=None):
    proba = np.asarray(proba)
    y_pred = proba.argmax(axis=1)
    rep = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "top3": top_k_accuracy(proba, y_true, 3),
        "mean_confidence": float(proba.max(axis=1).mean()),
        "ece": expected_calibration_error(proba, y_true),
        "n_test": int(len(y_true)),
    }
    if n_classes:
        rep["chance"] = 1.0 / n_classes
    if extra:
        rep.update(extra)
    return rep


# ------------------------------------------------------------------ regression
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot else float("nan")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rpd(y_true, y_pred):
    sd = float(np.std(np.asarray(y_true, float), ddof=1))
    r = rmse(y_true, y_pred)
    return float(sd / r) if r else float("nan")


def rpiq(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    iqr = float(np.percentile(y_true, 75) - np.percentile(y_true, 25))
    r = rmse(y_true, y_pred)
    return float(iqr / r) if r else float("nan")


def bias(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred, float) - np.asarray(y_true, float)))


def rpd_label(v):
    if v != v:
        return "n/a"
    if v >= 2.0:
        return "good"
    if v >= 1.5:
        return "fair"
    return "poor"


def regression_report(y_true, y_pred, extra=None):
    rep = {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rpd": rpd(y_true, y_pred),
        "rpd_label": rpd_label(rpd(y_true, y_pred)),
        "rpiq": rpiq(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "y_mean": float(np.mean(y_true)),
        "y_sd": float(np.std(y_true, ddof=1)),
        "n_test": int(len(y_true)),
    }
    if extra:
        rep.update(extra)
    return rep
