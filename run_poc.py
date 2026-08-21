#!/usr/bin/env python3
"""Train MobileNetV2 on CytoPacq public data + complementary frames."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier import METRICS_PATH, MODEL_PATH, load_model, predict, train

EXAMPLES = [
    ("01-fluorescence.jpg", "fluorescence", "Fluorescence"),
    ("02-brightfield.jpg", "brightfield", "Fond clair"),
    ("03-phase-contrast.jpg", "phase_contrast", "Phase"),
    ("04-sem.jpg", "sem", "MEB"),
    ("05-darkfield.jpg", "darkfield", "Fond noir"),
    ("06-confocal.jpg", "confocal", "Confocal"),
    ("07-fluorescence-2.jpg", "fluorescence", "Fluo fibroblastes"),
    ("08-brightfield-2.jpg", "brightfield", "H&E dense"),
]


def main() -> None:
    print("==> NOW medical · Panopticon (CytoPacq + MobileNetV2)", flush=True)
    metrics = train(
        per_class=48,
        seed=11,
        epochs=8,
        use_cytopacq=True,
        download_cytopacq=False,
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"    train={metrics['n_train']} test={metrics['n_test']}", flush=True)
    print(
        f"    cytopacq={metrics['dataset']['cytopacq']} "
        f"synth_fill={metrics['dataset']['synth_fill']} "
        f"real_aug={metrics['dataset'].get('real_aug', 0)}",
        flush=True,
    )
    print(f"    hold-out accuracy={metrics['accuracy']:.1%}", flush=True)
    for name, f1 in metrics["per_class_f1"].items():
        print(f"    {name:18s} f1={f1:.2f}", flush=True)
    print(f"    model → {MODEL_PATH}", flush=True)

    model = load_model()
    real: dict[str, str] = {}
    print("    real Wikimedia examples:", flush=True)
    for filename, gt, caption in EXAMPLES:
        path = ROOT / "data" / "examples" / filename
        if not path.exists():
            continue
        rgb = np.array(Image.open(path).convert("RGB"))
        pred = predict(rgb, model=model)
        mark = "ok" if pred["label"] == gt else "miss"
        real[caption] = f"{mark} {pred['label']} {pred['confidence']:.2f}"
        print(f"    {caption:18s} {mark} → {pred['label']} ({pred['confidence']:.0%})", flush=True)
    metrics["real_examples"] = real
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    metrics → {METRICS_PATH}", flush=True)
    assert metrics["accuracy"] >= 0.80, f"accuracy too low: {metrics['accuracy']}"
    print("PANOPTICON POC OK", flush=True)


if __name__ == "__main__":
    main()
