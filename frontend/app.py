#!/usr/bin/env python3
"""NOW medical — The Crimes of Microscopes and Their Panopticon."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classes import BLURBS, LABELS_FR
from src.classifier import load_model, predict
from src.features import FEATURE_NAMES, extract_features

st.set_page_config(
    page_title="Panopticon · NOW medical",
    page_icon="◉",
    layout="wide",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Instrument+Serif:ital@0;1&family=Syne:wght@700;800&display=swap');
      .block-container { max-width: 1180px; padding-top: 1.1rem; }
      h1, .hero-title {
        font-family: 'Syne', sans-serif;
        letter-spacing: -0.04em;
        font-weight: 800;
      }
      .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #d4a017;
        margin-bottom: 0.2rem;
      }
      .sub {
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        color: #b7b3a8;
        font-size: 1.15rem;
        margin-bottom: 1.2rem;
      }
      .verdict {
        font-family: 'Syne', sans-serif;
        font-size: 1.85rem;
        line-height: 1.1;
        color: #3ee0c0;
      }
      .hint { color: #8c887c; font-size: 0.92rem; }
      div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
      .ex-help { color: #8c887c; font-size: 0.88rem; margin: 0 0 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">NOW medical · unit 08</div>', unsafe_allow_html=True)
st.title("The Crimes of Microscopes and Their Panopticon")
st.markdown(
    '<p class="sub">Identifier le type de microscope à partir d’images brutes uniquement.</p>',
    unsafe_allow_html=True,
)

EXAMPLES_DIR = ROOT / "data" / "examples"
# Vraies frames (Wikimedia) — clic = saisie auto.
EXAMPLES: list[tuple[str, str, str]] = [
    ("01-fluorescence.jpg", "fluorescence", "Fluorescence"),
    ("02-brightfield.jpg", "brightfield", "Fond clair"),
    ("03-phase-contrast.jpg", "phase_contrast", "Phase"),
    ("04-sem.jpg", "sem", "MEB"),
    ("05-darkfield.jpg", "darkfield", "Fond noir"),
    ("06-confocal.jpg", "confocal", "Confocal"),
    ("07-fluorescence-2.jpg", "fluorescence", "Fluo fibroblastes"),
    ("08-brightfield-2.jpg", "brightfield", "H&E dense"),
]


@st.cache_resource(show_spinner=False)
def _model():
    return load_model()


@st.cache_data(show_spinner=False)
def _example_rgb(filename: str) -> np.ndarray:
    return np.array(Image.open(EXAMPLES_DIR / filename).convert("RGB"))


def _pick_example(i: int) -> None:
    st.session_state.example_i = i
    st.session_state.use_upload = False


def _on_upload() -> None:
    if st.session_state.get("uploader") is not None:
        st.session_state.use_upload = True


def _load_upload(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))


if "example_i" not in st.session_state:
    st.session_state.example_i = 0
if "use_upload" not in st.session_state:
    st.session_state.use_upload = False

model = _model()

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader(
        "JPG / PNG microscope",
        type=["jpg", "jpeg", "png"],
        key="uploader",
        on_change=_on_upload,
    )
    st.caption(
        "Ou clique un exemple à droite. Le modèle est entraîné sur du synthétique "
        "— une vraie photo peut être incertaine."
    )

st.markdown('<div class="eyebrow">Exemples</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="ex-help">Photos réelles. Clique un exemple pour le saisir automatiquement.</p>',
    unsafe_allow_html=True,
)

picked = int(st.session_state.example_i)
using_upload = bool(st.session_state.use_upload and uploaded is not None)

for row_start in (0, 4):
    cols = st.columns(4, gap="small")
    for j, col in enumerate(cols):
        i = row_start + j
        filename, _label, caption = EXAMPLES[i]
        selected = (not using_upload) and picked == i
        with col:
            with st.container(border=True):
                st.image(_example_rgb(filename), use_container_width=True)
                st.button(
                    caption,
                    key=f"ex-{i}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                    on_click=_pick_example,
                    args=(i,),
                    help="Saisir cette frame et lancer l’analyse",
                )

if using_upload:
    rgb = _load_upload(uploaded)
    gt = None
    source_caption = "Upload"
else:
    filename, label, caption = EXAMPLES[picked]
    rgb = _example_rgb(filename)
    gt = label
    source_caption = f"Exemple · {caption}"

st.divider()

result = predict(rgb, model=model)
feats = extract_features(rgb)

left, right = st.columns((1.15, 1), gap="large")
with left:
    st.image(rgb, caption=f"Frame brute · {source_caption}", use_container_width=True)
with right:
    st.markdown('<div class="eyebrow">Verdict</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="verdict">{LABELS_FR[result["label"]]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="hint">{BLURBS[result["label"]]}</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Confiance", f"{result['confidence']:.0%}")
    if gt is not None:
        ok = result["label"] == gt
        c2.metric("Contrôle", "match" if ok else "miss")

    st.subheader("Classement")
    for row in result["ranking"]:
        st.progress(min(max(row["score"], 0.0), 1.0), text=f"{LABELS_FR[row['label']]}  {row['score']:.0%}")

with st.expander("Preuves — signatures visuelles"):
    cols = st.columns(4)
    show = {
        "lum_mean": feats[0],
        "sat_mean": feats[2],
        "dark_frac": feats[3],
        "grayness": feats[11],
        "edge_density": feats[12],
        "pink_score": feats[9],
        "chroma": feats[10],
        "sparse_hot": feats[13],
    }
    for i, (name, val) in enumerate(show.items()):
        cols[i % 4].metric(name, f"{val:.2f}")
    st.caption("Features: " + ", ".join(FEATURE_NAMES))

with st.expander("Sources des exemples"):
    st.caption(
        "Photos réelles, Wikimedia Commons (redimensionnées). "
        "Licences : domaine public / CC0 / CC BY-SA. Détail : data/examples/SOURCES.txt"
    )
    st.markdown(
        "- Fluorescence — BPAE, CC BY-SA 3.0\n"
        "- Fond clair — prostate H&E, Mikael Häggström, CC0\n"
        "- Phase — cellules CHO, domaine public\n"
        "- MEB — caillot, Janice Carr / CDC, domaine public\n"
        "- Fond noir — GR, Dr Graham Beards, CC BY-SA 4.0\n"
        "- Confocal — HeLa, CC BY-SA 4.0\n"
        "- Fluo fibroblastes — Faust & Capco / NIH, domaine public\n"
        "- H&E dense — adénocarcinome, domaine public"
    )
