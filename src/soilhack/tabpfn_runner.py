"""Explicit TabPFN-3.5 model factory.

Rule (from the canonical TabPFN skill): NAME THE CHECKPOINT EXPLICITLY. Upstream has
changed the package default between releases, so a bare TabPFNRegressor() loads whatever
the installed package happens to default to. This module always pins V3.5.
"""
import os

# Enable the built-model cache BEFORE importing tabpfn (ships OFF; large gain for sweeps
# that rebuild an estimator many times). Harmless for a single fit; verified identical.
os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "4")

import numpy as np  # noqa: E402

MODEL_VERSION = "v3.5"          # the hackathon target
DEFAULT_DEVICE = "cuda"         # falls back to cpu below if CUDA is absent


def resolved_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = resolved_device()


def describe_version():
    """Return a dict describing the actual model the install offers for V3.5."""
    import tabpfn
    info = {"tabpfn_version": getattr(tabpfn, "__version__", "?")}
    try:
        from tabpfn import model_loading as ml
        src = ml._get_model_source(ml.ModelVersion.V3_5, ml.ModelType.REGRESSOR)
        info["regressor_repo"] = src.repo_id
        info["regressor_filename"] = src.default_filename
    except Exception as e:  # pragma: no cover
        info["model_source_err"] = repr(e)
    info["device"] = DEVICE
    return info


def make_classifier(**kw):
    """TabPFN-3.5 classifier, pinned. Explicit n_estimators='auto' to cover wide features."""
    from tabpfn import TabPFNClassifier
    kw.setdefault("device", DEVICE)
    return TabPFNClassifier.create_default_for_version(MODEL_VERSION, **kw)


def make_regressor(**kw):
    """TabPFN-3.5 regressor, pinned."""
    from tabpfn import TabPFNRegressor
    kw.setdefault("device", DEVICE)
    return TabPFNRegressor.create_default_for_version(MODEL_VERSION, **kw)


def predict_class_proba(model, X_query):
    p = model.predict_proba(X_query)
    return np.asarray(p)


def predict_regression(model, X_query):
    """Return point predictions; handles the dict/quantile return shape if present."""
    out = model.predict(X_query)
    if isinstance(out, dict):
        # some builds return {'mean':..., 'quantiles':...}
        for key in ("mean", "median", "prediction"):
            if key in out:
                return np.asarray(out[key])
        # fall back to the first value
        return np.asarray(next(iter(out.values())))
    return np.asarray(out)
