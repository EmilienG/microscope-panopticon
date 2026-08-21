"""Hand-crafted visual signatures of common microscope modalities."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "lum_mean",
    "lum_std",
    "sat_mean",
    "dark_frac",
    "bright_frac",
    "r_mean",
    "g_mean",
    "b_mean",
    "g_over_r",
    "pink_score",
    "chroma",
    "grayness",
    "edge_density",
    "sparse_hot",
)


def extract_features(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB image")
    x = rgb.astype(np.float32)
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-6)
    chroma = (np.std(x, axis=2)).mean()
    grayness = 1.0 - (np.abs(r - g) + np.abs(g - b) + np.abs(b - r)).mean() / (3 * 255.0)
    gy, gx = np.gradient(lum)
    edge = np.hypot(gx, gy).mean()
    hot = lum > 80
    sparse_hot = float(hot.mean()) if lum.mean() < 80 else 0.0

    feats = np.array(
        [
            float(lum.mean()),
            float(lum.std()),
            float(sat.mean()),
            float((lum < 40).mean()),
            float((lum > 200).mean()),
            float(r.mean()),
            float(g.mean()),
            float(b.mean()),
            float(g.mean() / (r.mean() + 1e-3)),
            float((r - g).mean()),
            float(chroma),
            float(grayness),
            float(edge),
            sparse_hot,
        ],
        dtype=np.float32,
    )
    return feats


def extract_batch(images: list[np.ndarray]) -> np.ndarray:
    return np.stack([extract_features(im) for im in images], axis=0)
