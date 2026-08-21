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

from src.classes import BLURBS, CLASSES, LABELS_FR
from src.classifier import load_model, predict
from src.features import FEATURE_NAMES, extract_features
from src.synth import make_image

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


@st.cache_resource(show_spinner=False)
def _model():
    return load_model()


@st.cache_data(show_spinner=False)
def _demo(label: str, seed: int) -> np.ndarray:
    return make_image(label, seed=seed)


model = _model()

with st.sidebar:
    st.header("Source")
    mode = st.radio("Image", ["Galerie synthétique", "Upload"], index=0)
    label = st.selectbox(
        "Modalité (démo)",
        list(CLASSES),
        format_func=lambda k: LABELS_FR[k],
        index=0,
    )
    seed = st.slider("Seed", 0, 99, 3)
    uploaded = None
    if mode == "Upload":
        uploaded = st.file_uploader("JPG / PNG microscope", type=["jpg", "jpeg", "png"])
        st.caption("Le modèle a été entraîné sur des frames synthétiques. Une vraie photo peut être incertaine — c’est le point du PoC.")


def _load_upload(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))


rgb: np.ndarray | None = None
gt: str | None = None
if mode == "Galerie synthétique":
    rgb = _demo(label, seed)
    gt = label
elif uploaded is not None:
    rgb = _load_upload(uploaded)

if rgb is None:
    st.info("Choisis une vue synthétique ou charge une image.")
    st.stop()

result = predict(rgb, model=model)
feats = extract_features(rgb)

left, right = st.columns((1.15, 1), gap="large")
with left:
    st.image(rgb, caption="Frame brute", use_container_width=True)
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
