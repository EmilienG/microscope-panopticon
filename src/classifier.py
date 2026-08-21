"""Tiny RandomForest over microscope visual signatures."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classes import CLASSES
from src.features import extract_batch, extract_features
from src.synth import make_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "modality_clf.joblib"
METRICS_PATH = ROOT / "data" / "processed" / "metrics.json"


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(
    per_class: int = 50,
    seed: int = 0,
    model_path: Path | None = None,
) -> dict:
    images, labels = make_dataset(per_class=per_class, seed=seed)
    x = extract_batch(images)
    y = np.array(labels)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, random_state=7, stratify=y
    )
    clf = _pipeline()
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    acc = float(accuracy_score(y_te, pred))
    report = classification_report(y_te, pred, labels=list(CLASSES), output_dict=True, zero_division=0)

    path = model_path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)

    per_class_acc = {
        name: float(report[name]["f1-score"])
        for name in CLASSES
        if name in report
    }
    return {
        "accuracy": round(acc, 4),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "per_class_f1": {k: round(v, 3) for k, v in per_class_acc.items()},
        "model": str(path),
        "classes": list(CLASSES),
    }


def load_model(model_path: Path | None = None) -> Pipeline:
    path = model_path or MODEL_PATH
    if not path.exists():
        train(model_path=path)
    return joblib.load(path)


def predict_proba(rgb: np.ndarray, model: Pipeline | None = None) -> dict[str, float]:
    clf = model or load_model()
    feats = extract_features(rgb).reshape(1, -1)
    classes = list(clf.classes_)
    probs = clf.predict_proba(feats)[0]
    return {c: float(p) for c, p in zip(classes, probs, strict=True)}


def predict(rgb: np.ndarray, model: Pipeline | None = None) -> dict:
    probs = predict_proba(rgb, model=model)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, conf = ranked[0]
    return {
        "label": top,
        "confidence": round(conf, 4),
        "ranking": [{"label": k, "score": round(v, 4)} for k, v in ranked],
        "probs": {k: round(v, 4) for k, v in probs.items()},
    }
