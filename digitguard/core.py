"""Probability calibration, conformal sets and inference policy."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

ARTIFACT_VERSION = 1
TARGET_COVERAGE = 0.90
TARGET_ACCURACY = 0.97
MIN_ACCEPTED = 30


def calibrated(probabilities, temperature):
    return softmax(np.log(np.clip(probabilities, 1e-12, 1)) / temperature, axis=1)


def fit_temperature(probabilities, labels):
    def loss(temperature):
        p = calibrated(probabilities, temperature)
        return -np.log(p[np.arange(len(labels)), labels]).mean()

    result = minimize_scalar(loss, bounds=(0.1, 10), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return float(result.x)


def conformal_cutoff(probabilities, labels, coverage=TARGET_COVERAGE):
    scores = 1 - probabilities[np.arange(len(labels)), labels]
    rank = int(np.ceil((len(scores) + 1) * coverage))
    return float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else 1.0


def prediction_sets(probabilities, cutoff):
    return (1 - probabilities) <= cutoff


def select_threshold(probabilities, labels, cutoff):
    """Find the highest-coverage policy meeting the validation accuracy target."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    singleton = prediction_sets(probabilities, cutoff).sum(axis=1) == 1
    for threshold in np.unique(confidence[singleton]):
        accepted = singleton & (confidence >= threshold)
        if accepted.sum() >= MIN_ACCEPTED and correct[accepted].mean() >= TARGET_ACCURACY:
            return float(threshold)
    return 1.01  # Abstain on every row if no eligible threshold exists.


def infer(bundle, pixels):
    pixels = np.asarray(pixels, dtype=float)
    if pixels.shape != (64,) or not np.isfinite(pixels).all() or ((pixels < 0) | (pixels > 16)).any():
        raise ValueError("Provide exactly 64 finite pixel values between 0 and 16")
    if not pixels.any():
        return {"prediction": None, "confidence": None, "candidates": [],
                "probabilities": None, "reason": "blank_input"}
    probabilities = calibrated(bundle["model"].predict_proba(pixels[None]), bundle["temperature"])
    p = probabilities[0]
    candidates = np.flatnonzero(prediction_sets(probabilities, bundle["cutoff"])[0]).tolist()
    confidence = float(p.max())
    accepted = len(candidates) == 1 and confidence >= bundle["threshold"]
    return {
        "prediction": int(p.argmax()) if accepted else None,
        "confidence": confidence,
        "candidates": candidates,
        "probabilities": p.tolist(),
        "reason": "accepted" if accepted else "uncertain",
    }
