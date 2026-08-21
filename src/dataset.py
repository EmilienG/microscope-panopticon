"""Assemble the training set: CytoPacq public volumes + complementary synth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.classes import CLASSES
from src.cytopacq import cytopacq_frames
from src.synth import make_image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "data" / "examples"
REAL_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("01-fluorescence.jpg", "fluorescence"),
    ("02-brightfield.jpg", "brightfield"),
    ("03-phase-contrast.jpg", "phase_contrast"),
    ("04-sem.jpg", "sem"),
    ("05-darkfield.jpg", "darkfield"),
    ("06-confocal.jpg", "confocal"),
    ("07-fluorescence-2.jpg", "fluorescence"),
    ("08-brightfield-2.jpg", "brightfield"),
)


def load_real_examples(*, copies: int = 8, seed: int = 0) -> tuple[list[np.ndarray], list[str]]:
    """Wikimedia demo frames + flips/rotations (kept out of the synth hold-out)."""
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[str] = []
    for filename, label in REAL_EXAMPLES:
        path = EXAMPLES_DIR / filename
        if not path.exists():
            continue
        base = np.array(Image.open(path).convert("RGB"))
        pil = Image.fromarray(base)
        variants = [base]
        for _ in range(max(0, copies - 1)):
            im = pil
            if rng.random() > 0.5:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() > 0.5:
                im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            angle = int(rng.choice([0, 90, 180, 270]))
            if angle:
                im = im.rotate(angle, expand=False)
            variants.append(np.array(im.convert("RGB")))
        images.extend(variants)
        labels.extend([label] * len(variants))
    return images, labels


def build_dataset(
    per_class: int = 64,
    seed: int = 0,
    *,
    use_cytopacq: bool = True,
    download_cytopacq: bool = False,
    max_cytopacq: int = 120,
) -> tuple[list[np.ndarray], list[str], dict[str, int]]:
    """Return images, labels, and per-source counts.

    CytoPacq (experimental PSF, Zeiss/Atto CARV + CTC A549-SIM) fills
    confocal / fluorescence first; remaining slots use the local generator
    so every modality has ``per_class`` samples.
    """
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    source_counts = {"cytopacq": 0, "synth": 0}

    if use_cytopacq:
        imgs, labs = cytopacq_frames(max_a549=max_cytopacq, download=download_cytopacq)
        for img, lab in zip(imgs, labs, strict=True):
            if lab in buckets and len(buckets[lab]) < per_class:
                buckets[lab].append(img)
                source_counts["cytopacq"] += 1

    for label in CLASSES:
        while len(buckets[label]) < per_class:
            child = int(rng.integers(0, 2**31 - 1))
            buckets[label].append(make_image(label, seed=child))
            source_counts["synth"] += 1

    images: list[np.ndarray] = []
    labels: list[str] = []
    for label in CLASSES:
        images.extend(buckets[label][:per_class])
        labels.extend([label] * per_class)
    return images, labels, source_counts
