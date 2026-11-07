"""Reproducible training and evaluation from the bundled UCI digits data."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.ndimage import shift
from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .core import (
    ARTIFACT_VERSION, calibrated, conformal_cutoff, fit_temperature,
    prediction_sets, select_threshold,
)

SEED = 42


def split_indices(labels):
    """60% model search, 10% temperature, 10% conformal, 10% policy, 10% test."""
    fit, rest = train_test_split(
        np.arange(len(labels)), test_size=0.4, stratify=labels, random_state=SEED
    )
    temperature, rest = train_test_split(
        rest, test_size=0.75, stratify=labels[rest], random_state=SEED + 1
    )
    conformal, rest = train_test_split(
        rest, test_size=2 / 3, stratify=labels[rest], random_state=SEED + 2
    )
    policy, test = train_test_split(
        rest, test_size=0.5, stratify=labels[rest], random_state=SEED + 3
    )
    return dict(fit=fit, temperature=temperature, conformal=conformal, policy=policy, test=test)


def evaluate(probabilities, labels, cutoff, threshold):
    predicted = probabilities.argmax(axis=1)
    sets = prediction_sets(probabilities, cutoff)
    accepted = (sets.sum(axis=1) == 1) & (probabilities.max(axis=1) >= threshold)
    return {
        "rows": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro"),
        "log_loss": log_loss(labels, probabilities, labels=np.arange(10)),
        "conformal_coverage": float(sets[np.arange(len(labels)), labels].mean()),
        "mean_set_size": float(sets.sum(axis=1).mean()),
        "accepted": int(accepted.sum()),
        "acceptance_rate": float(accepted.mean()),
        "accepted_accuracy": (
            accuracy_score(labels[accepted], predicted[accepted]) if accepted.any() else None
        ),
    }


def stress_cases(x):
    rng = np.random.default_rng(SEED)
    shifted = shift(x.reshape(-1, 8, 8), (0, 0, 1), order=0, mode="constant").reshape(-1, 64)
    occluded = x.reshape(-1, 8, 8).copy()
    occluded[:, 3:5, 3:5] = 0
    return {
        "gaussian_noise_sd_3": np.clip(x + rng.normal(0, 3, x.shape), 0, 16),
        "shift_right_1px": shifted,
        "center_occlusion_2x2": occluded.reshape(-1, 64),
    }


def train(model_path, report_dir):
    x, y = load_digits(return_X_y=True)
    indices = split_indices(y)
    fit, temp, conformal, policy, test = indices.values()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = {
        "logistic": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
            {"logisticregression__C": [0.1, 1, 10]},
        ),
        "extra_trees": (
            ExtraTreesClassifier(n_estimators=160, random_state=SEED, n_jobs=1),
            {"max_features": ["sqrt", 0.6], "min_samples_leaf": [1, 2]},
        ),
        "rbf_svm": (
            make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
            {"svc__C": [3, 10, 30], "svc__gamma": ["scale", 0.02]},
        ),
    }
    searches = {}
    leaderboard = {}
    for name, (estimator, grid) in models.items():
        search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, error_score="raise")
        search.fit(x[fit], y[fit])
        searches[name] = search
        leaderboard[name] = {
            "cv_macro_f1": search.best_score_,
            "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "parameters": search.best_params_,
        }
        print(f"{name:12} CV macro F1 = {search.best_score_:.4f}")
    winner = max(searches, key=lambda name: searches[name].best_score_)
    model = searches[winner].best_estimator_
    temperature = fit_temperature(model.predict_proba(x[temp]), y[temp])

    def predict(values):
        return calibrated(model.predict_proba(values), temperature)

    cutoff = conformal_cutoff(predict(x[conformal]), y[conformal])
    threshold = select_threshold(predict(x[policy]), y[policy], cutoff)
    p_test = predict(x[test])
    report = {
        "artifact_version": ARTIFACT_VERSION,
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "splits": {name: len(rows) for name, rows in indices.items()},
        "leaderboard": leaderboard,
        "winner": winner,
        "temperature": temperature,
        "conformal_cutoff": cutoff,
        "confidence_threshold": threshold,
        "policy_validation": evaluate(predict(x[policy]), y[policy], cutoff, threshold),
        "test": evaluate(p_test, y[test], cutoff, threshold),
        "uncalibrated_test_log_loss": log_loss(y[test], model.predict_proba(x[test])),
        "confusion_matrix": confusion_matrix(y[test], p_test.argmax(axis=1), labels=np.arange(10)).tolist(),
        "stress": {
            name: evaluate(predict(values), y[test], cutoff, threshold)
            for name, values in stress_cases(x[test]).items()
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "version": ARTIFACT_VERSION,
        "sklearn_version": sklearn.__version__,
        "model": model,
        "name": winner,
        "temperature": temperature,
        "cutoff": cutoff,
        "threshold": threshold,
    }, model_path, compress=3)
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {winner}. Test accuracy = {report['test']['accuracy']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    options = parser.parse_args()
    train(options.model, options.report_dir)


if __name__ == "__main__":
    main()
