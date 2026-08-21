"""Tests Panopticon — features, synth, classifieur."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classes import CLASSES
from src.classifier import predict, train
from src.features import FEATURE_NAMES, extract_features
from src.synth import make_dataset, make_image


def test_feature_vector_shape():
    img = make_image("fluorescence", seed=1)
    feats = extract_features(img)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(feats).all()


def test_synth_all_classes():
    for label in CLASSES:
        img = make_image(label, seed=2)
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.uint8


def test_dataset_counts():
    images, labels = make_dataset(per_class=2, seed=0)
    assert len(images) == len(labels) == 2 * len(CLASSES)
    assert set(labels) == set(CLASSES)


def test_train_and_predict(tmp_path):
    model_path = tmp_path / "clf.joblib"
    metrics = train(per_class=24, seed=3, model_path=model_path)
    assert metrics["accuracy"] >= 0.85
    assert model_path.exists()

    from joblib import load

    clf = load(model_path)
    img = make_image("brightfield", seed=9)
    out = predict(img, model=clf)
    assert out["label"] in CLASSES
    assert 0.0 <= out["confidence"] <= 1.0
    assert len(out["ranking"]) == len(CLASSES)
