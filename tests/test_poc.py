"""Tests Panopticon — features, CytoPacq loader, CNN classifier."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classes import CLASSES
from src.classifier import predict, train
from src.cytopacq import load_mucic_previews
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


def test_cytopacq_previews_are_rgb():
    frames = load_mucic_previews()
    assert len(frames) >= 8
    assert frames[0].ndim == 3 and frames[0].shape[2] == 3


def test_train_and_predict(tmp_path):
    model_path = tmp_path / "clf.onnx"
    metrics = train(
        per_class=8,
        seed=3,
        model_path=model_path,
        epochs=1,
        batch_size=8,
        use_cytopacq=False,
    )
    assert metrics["accuracy"] >= 0.4
    assert model_path.exists()

    from src.classifier import load_model

    clf = load_model(model_path)
    img = make_image("brightfield", seed=9)
    out = predict(img, model=clf)
    assert out["label"] in CLASSES
    assert 0.0 <= out["confidence"] <= 1.0
    assert len(out["ranking"]) == len(CLASSES)


def test_predict_ignores_legacy_sklearn_model():
    """Stale Streamlit cache used to pass a StandardScaler pipeline."""

    class LegacyPipeline:
        def predict_proba(self, x):
            raise AssertionError("sklearn path must not run")

    rgb = np.array(
        Image.open(ROOT / "data" / "examples" / "04-sem.jpg").convert("RGB")
    )
    out = predict(rgb, model=LegacyPipeline())
    assert out["label"] in CLASSES
    assert len(out["ranking"]) == len(CLASSES)
