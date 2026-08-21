"""CytoPacq public data — experimental-PSF fluorescence / confocal frames.

Sources:
- Simulator: https://cbia.fi.muni.cz/simulator/index.php
- MUCIC HL60 previews (Zeiss S100 + Atto CARV confocal)
- CTC Fluo-C3DH-A549-SIM (FiloGen / CytoPacq, GFP-actin)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PREVIEWS_DIR = ROOT / "data" / "cytopacq" / "previews"
RAW_DIR = ROOT / "data" / "raw" / "cytopacq"
A549_ZIP = RAW_DIR / "Fluo-C3DH-A549-SIM.zip"
A549_URL = "https://data.celltrackingchallenge.net/training-datasets/Fluo-C3DH-A549-SIM.zip"
SLICES_DIR = RAW_DIR / "a549_slices"

SIMULATOR_URL = "https://cbia.fi.muni.cz/simulator/index.php"
MUCIC_URL = "https://cbia.fi.muni.cz/datasets/"


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        g = arr
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return np.asarray(Image.fromarray(arr[..., :3].astype(np.uint8)).convert("RGB"))
    else:
        g = np.asarray(arr)
        while g.ndim > 2:
            g = g[g.shape[0] // 2]
    g = np.asarray(g, dtype=np.float32)
    if g.max() > 0:
        g = g / g.max() * 255.0
    rgb = np.stack([g, g, g], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _crop_xy_panel(img: Image.Image) -> Image.Image:
    """MUCIC previews are orthogonal viewers: keep the large XY pane."""
    w, h = img.size
    return img.crop((0, 0, int(w * 0.72), int(h * 0.72)))


def load_mucic_previews() -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    if not PREVIEWS_DIR.exists():
        return frames
    for path in sorted(PREVIEWS_DIR.glob("*.png")):
        img = _crop_xy_panel(Image.open(path).convert("RGB"))
        frames.append(np.array(img))
    return frames


def download_a549(timeout: int = 600) -> Path:
    """Fetch the CTC CytoPacq A549 archive if missing."""
    import urllib.request

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if A549_ZIP.exists() and A549_ZIP.stat().st_size > 1_000_000:
        return A549_ZIP
    req = urllib.request.Request(A549_URL, headers={"User-Agent": "PanopticonPoC/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, A549_ZIP.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
    return A549_ZIP


def _read_tiff_bytes(raw: bytes) -> np.ndarray:
    """LZW stacks from CTC/CytoPacq: Pillow, not tifffile (needs imagecodecs)."""
    im = Image.open(io.BytesIO(raw))
    n = int(getattr(im, "n_frames", 1) or 1)
    frames = []
    for i in range(n):
        im.seek(i)
        frames.append(np.array(im))
    if len(frames) == 1:
        return frames[0]
    return np.stack(frames, axis=0)


def _mid_slices(vol: np.ndarray, n: int = 3) -> list[np.ndarray]:
    v = np.asarray(vol)
    if v.ndim == 2:
        return [_to_rgb(v)]
    # drop trailing channel axis
    if v.ndim >= 3 and v.shape[-1] in (1, 3, 4) and v.shape[0] > 8:
        zstack = v
    elif v.ndim >= 3:
        zstack = v
    else:
        return [_to_rgb(v)]
    z = zstack.shape[0]
    if z <= n:
        idxs = list(range(z))
    else:
        lo, hi = max(1, int(z * 0.3)), min(z - 1, int(z * 0.7))
        idxs = np.linspace(lo, hi, n).astype(int).tolist()
    return [_to_rgb(zstack[i]) for i in idxs]


def extract_a549_slices(max_images: int = 120, per_file: int = 2) -> list[np.ndarray]:
    """2D frames from CytoPacq A549-SIM (skip GT folders). Cached as JPEG."""
    SLICES_DIR.mkdir(parents=True, exist_ok=True)
    cached = sorted(SLICES_DIR.glob("*.jpg"))
    if len(cached) >= max_images:
        return [np.array(Image.open(p).convert("RGB")) for p in cached[:max_images]]
    if not A549_ZIP.exists() or A549_ZIP.stat().st_size < 1_000_000:
        return [np.array(Image.open(p).convert("RGB")) for p in cached[:max_images]]

    frames: list[np.ndarray] = []
    with zipfile.ZipFile(A549_ZIP) as zf:
        names = [
            n
            for n in zf.namelist()
            if n.lower().endswith((".tif", ".tiff"))
            and "_GT" not in n
            and "/TRA/" not in n
            and not n.startswith("__")
        ]
        names.sort()
        step = max(1, len(names) // max(1, max_images // per_file))
        for name in names[::step]:
            if len(frames) >= max_images:
                break
            try:
                vol = _read_tiff_bytes(zf.read(name))
            except Exception:
                continue
            for sl in _mid_slices(vol, n=per_file):
                frames.append(sl)
                if len(frames) >= max_images:
                    break
    for i, arr in enumerate(frames):
        Image.fromarray(arr).save(SLICES_DIR / f"a549_{i:04d}.jpg", quality=85)
    return frames


def cytopacq_frames(
    *,
    max_a549: int = 120,
    download: bool = False,
) -> tuple[list[np.ndarray], list[str]]:
    """Labeled RGB frames from official CytoPacq public datasets."""
    images: list[np.ndarray] = []
    labels: list[str] = []
    for img in load_mucic_previews():
        images.append(img)
        labels.append("confocal")
    if download and not (A549_ZIP.exists() and A549_ZIP.stat().st_size > 1_000_000):
        download_a549()
    for img in extract_a549_slices(max_images=max_a549):
        images.append(img)
        labels.append("fluorescence")
    return images, labels
