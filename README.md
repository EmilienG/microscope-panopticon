# Panopticon — NOW medical

PoC fonctionnel du pitch
[The Crimes of Microscopes and Their Panopticon](https://www.canva.com/design/DAHSvHKw92E/dqjuS2rd3jD_oyr3rmyAFw/edit) :
**identifier le type de microscope à partir d’images brutes uniquement.**

MobileNetV2 (poids ImageNet) affiné sur des volumes **CytoPacq**
([simulateur CBIA](https://cbia.fi.muni.cz/simulator/index.php), PSF expérimentales
Zeiss / Atto CARV) plus un complément local pour les modalités que CytoPacq
ne couvre pas (H&E, phase, MEB, fond noir).

## Pitch (20 s)

Une frame brute entre. Le Panopticon dit si c’est de la fluorescence, du fond
clair, du contraste de phase, du MEB, du fond noir ou du confocal — et montre
les preuves (signatures) qui ont trahi l’optique.

## Run

Env Le Wagon : `~/.pyenv/versions/lewagon/bin/python`

```bash
cd 08-microscope-panopticon
python run_poc.py    # CytoPacq + train MobileNetV2, hold-out ≥ 80 %
python run_ui.py     # http://localhost:8508
pytest -q tests
```

Le zip CTC `Fluo-C3DH-A549-SIM` (~330 Mo) se place dans `data/raw/cytopacq/`
s’il est déjà téléchargé ; sinon l’entraînement utilise les previews MUCIC
versionnées dans `data/cytopacq/previews/`.

## Deploy (Streamlit Community Cloud)

Le modèle `models/modality_mobilenet.onnx` est versionné (inference
`onnxruntime`, sans PyTorch).

1. Repo public : [`EmilienG/microscope-panopticon`](https://github.com/EmilienG/microscope-panopticon)
2. [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Branch `main`
4. **Main file path** : `frontend/app.py`
5. **Python version** : 3.11 ou 3.12
6. Deploy

## Pipeline

1. Dataset CytoPacq (MUCIC HL60 confocal + A549-SIM GFP) — `src/cytopacq.py`
2. Complément de frames par modalité — `src/synth.py` / `src/dataset.py`
3. MobileNetV2 → ONNX — `src/classifier.py`
4. UI Streamlit (8 photos réelles cliquables ou upload) — `frontend/app.py`

## Classes

| id | UI | Dataset visé |
|---|---|---|
| `fluorescence` | Fluorescence | CytoPacq A549-SIM (GFP) |
| `brightfield` | Fond clair (H&E) | complément local |
| `phase_contrast` | Contraste de phase | complément local |
| `sem` | MEB / SEM | complément local |
| `darkfield` | Fond noir | complément local |
| `confocal` | Confocal | CytoPacq MUCIC HL60 (Atto CARV) |

## Limites (assumées)

CytoPacq ne simule que de la fluorescence (widefield / confocal). Les autres
optiques sont comblées par un générateur local : le PoC montre le pipeline
image brute → modalité, pas un modèle clinique.
