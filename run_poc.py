#!/usr/bin/env python3
"""Panopticon end-to-end: synth frames → train → hold-out accuracy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier import METRICS_PATH, train
from src.synth import make_image

DEMO_DIR = ROOT / "data" / "processed"


def main() -> None:
    print("==> NOW medical · Panopticon", flush=True)
    metrics = train(per_class=48, seed=11)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    from PIL import Image

    for i, label in enumerate(metrics["classes"]):
        img = make_image(label, seed=100 + i)
        Image.fromarray(img).save(DEMO_DIR / f"demo_{label}.png")

    print(f"    train={metrics['n_train']} test={metrics['n_test']}", flush=True)
    print(f"    accuracy={metrics['accuracy']:.1%}", flush=True)
    for name, f1 in metrics["per_class_f1"].items():
        print(f"    {name:18s} f1={f1:.2f}", flush=True)
    print(f"    metrics → {METRICS_PATH}", flush=True)
    assert metrics["accuracy"] >= 0.90, f"accuracy too low: {metrics['accuracy']}"
    print("PANOPTICON POC OK", flush=True)


if __name__ == "__main__":
    main()
