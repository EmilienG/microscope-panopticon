#!/usr/bin/env python3
"""Raccourci: streamlit run frontend/app.py"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
app = ROOT / "frontend" / "app.py"
raise SystemExit(
    subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app),
            "--server.port",
            "8508",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(ROOT),
    )
)
