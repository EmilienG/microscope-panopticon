"""Synthetic microscope frames — distinctive enough to train a tiny model."""

from __future__ import annotations

import numpy as np

from src.classes import CLASSES

SIZE = 256


def _disk(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, r: float) -> np.ndarray:
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _gauss_blob(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, s: float) -> np.ndarray:
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s * s))


def _grid(n: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(n, dtype=np.float32)
    x = np.arange(n, dtype=np.float32)
    return np.meshgrid(y, x, indexing="ij")


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def make_fluorescence(rng: np.random.Generator, n: int = SIZE) -> np.ndarray:
    yy, xx = _grid(n)
    img = np.zeros((n, n, 3), dtype=np.float32)
    palettes = [
        (20, 230, 70),
        (40, 90, 240),
        (230, 40, 50),
        (40, 210, 210),
    ]
    k = int(rng.integers(4, 9))
    for _ in range(k):
        cy, cx = rng.uniform(20, n - 20, size=2)
        s = float(rng.uniform(8, 28))
        amp = float(rng.uniform(140, 255))
        r, g, b = palettes[int(rng.integers(0, len(palettes)))]
        blob = _gauss_blob(yy, xx, cy, cx, s)
        img[:, :, 0] += blob * amp * (r / 255.0)
        img[:, :, 1] += blob * amp * (g / 255.0)
        img[:, :, 2] += blob * amp * (b / 255.0)
    img += rng.normal(0, 3, img.shape).astype(np.float32)
    return _to_u8(img)


def make_brightfield(rng: np.random.Generator, n: int = SIZE) -> np.ndarray:
    yy, xx = _grid(n)
    cream = np.array([248, 242, 232], dtype=np.float32)
    img = np.broadcast_to(cream, (n, n, 3)).copy()
    k = int(rng.integers(5, 12))
    for _ in range(k):
        cy, cx = rng.uniform(15, n - 15, size=2)
        rx, ry = rng.uniform(12, 48), rng.uniform(10, 40)
        stain = rng.uniform(0.0, 1.0) > 0.45
        if stain:
            color = np.array([165, 45, 95], dtype=np.float32)  # eosin-ish
        else:
            color = np.array([95, 70, 145], dtype=np.float32)  # hematoxylin
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        alpha = float(rng.uniform(0.35, 0.75))
        img[mask] = (1 - alpha) * img[mask] + alpha * color
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    return _to_u8(img)


def make_phase_contrast(rng: np.random.Generator, n: int = SIZE) -> np.ndarray:
    yy, xx = _grid(n)
    bg = float(rng.uniform(118, 142))
    gray = np.full((n, n), bg, dtype=np.float32)
    k = int(rng.integers(6, 14))
    for _ in range(k):
        cy, cx = rng.uniform(18, n - 18, size=2)
        r = float(rng.uniform(8, 22))
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        cell = dist <= r
        halo = (dist > r) & (dist <= r * 1.35)
        gray[cell] = bg - float(rng.uniform(25, 55))
        gray[halo] = bg + float(rng.uniform(35, 70))
    gray += rng.normal(0, 5, gray.shape).astype(np.float32)
    img = np.stack([gray, gray, gray], axis=-1)
    return _to_u8(img)


def make_sem(rng: np.random.Generator, n: int = SIZE) -> np.ndarray:
    yy, xx = _grid(n)
    grain = rng.normal(110, 28, (n, n)).astype(np.float32)
    # low-frequency hills
    for _ in range(6):
        cy, cx = rng.uniform(0, n, size=2)
        s = float(rng.uniform(30, 90))
        grain += _gauss_blob(yy, xx, cy, cx, s) * float(rng.uniform(-40, 50))
    k = int(rng.integers(4, 10))
    for _ in range(k):
        cy, cx = rng.uniform(20, n - 20, size=2)
        r = float(rng.uniform(10, 32))
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        shade = np.clip(1.0 - dist / r, 0, 1)
        light = np.clip(1.2 - ((yy - (cy - r * 0.35)) ** 2 + (xx - (cx - r * 0.35)) ** 2) / (r * r), 0, 1)
        ball = dist <= r
        grain[ball] = 70 + 140 * light[ball] + 20 * shade[ball]
    grain += rng.normal(0, 8, grain.shape).astype(np.float32)
    img = np.stack([grain, grain, grain], axis=-1)
    return _to_u8(img)


def make_darkfield(rng: np.random.Generator, n: int = SIZE) -> np.ndarray:
    yy, xx = _grid(n)
    gray = rng.normal(4, 2, (n, n)).astype(np.float32)
    k = int(rng.integers(4, 9))
    for _ in range(k):
        cy, cx = rng.uniform(25, n - 25, size=2)
        r = float(rng.uniform(16, 50))
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        ring = np.abs(dist - r) < float(rng.uniform(1.2, 2.8))
        gray[ring] = float(rng.uniform(180, 255))
    gray += rng.normal(0, 3, gray.shape).astype(np.float32)
    img = np.stack([gray, gray * 0.98, gray * 0.95], axis=-1)
    return _to_u8(img)


def make_confocal(rng: np.random.Generator, n: int = SIZE) -> np.ndarray:
    img = np.zeros((n, n, 3), dtype=np.float32)
    channel = int(rng.integers(0, 3))
    n_dots = int(rng.integers(40, 90))
    ys = rng.integers(8, n - 8, size=n_dots)
    xs = rng.integers(8, n - 8, size=n_dots)
    for y, x in zip(ys, xs, strict=True):
        r = int(rng.integers(1, 3))
        img[y - r : y + r + 1, x - r : x + r + 1, channel] = float(rng.uniform(180, 255))
        if rng.random() > 0.7:
            other = (channel + 1) % 3
            img[y, x, other] = float(rng.uniform(40, 90))
    img += rng.normal(0, 2, img.shape).astype(np.float32)
    return _to_u8(img)


_MAKERS = {
    "fluorescence": make_fluorescence,
    "brightfield": make_brightfield,
    "phase_contrast": make_phase_contrast,
    "sem": make_sem,
    "darkfield": make_darkfield,
    "confocal": make_confocal,
}


def make_image(label: str, seed: int | None = None, n: int = SIZE) -> np.ndarray:
    if label not in _MAKERS:
        raise ValueError(f"unknown class: {label}")
    rng = np.random.default_rng(seed)
    return _MAKERS[label](rng, n=n)


def make_dataset(
    per_class: int = 40,
    seed: int = 0,
    n: int = SIZE,
) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[str] = []
    for label in CLASSES:
        for _ in range(per_class):
            child = int(rng.integers(0, 2**31 - 1))
            images.append(make_image(label, seed=child, n=n))
            labels.append(label)
    return images, labels
