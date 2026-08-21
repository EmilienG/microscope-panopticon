# Panopticon — NOW medical

PoC fonctionnel du pitch
[The Crimes of Microscopes and Their Panopticon](https://www.canva.com/design/DAHSvHKw92E/dqjuS2rd3jD_oyr3rmyAFw/edit) :
**identifier le type de microscope à partir d’images brutes uniquement.**

Pas de CNN, pas de poids à télécharger. Frames synthétiques → signatures visuelles
(luminance, saturation, fond noir, grain, rose H&E…) → RandomForest.

## Pitch (20 s)

Une frame brute entre. Le Panopticon dit si c’est de la fluorescence, du fond
clair, du contraste de phase, du MEB, du fond noir ou du confocal — et montre
les preuves (signatures) qui ont trahi l’optique.

## Run

Env Le Wagon : `~/.pyenv/versions/lewagon/bin/python`

```bash
cd 08-microscope-panopticon
python run_poc.py    # train + hold-out, attend accuracy ≥ 90 %
python run_ui.py     # http://localhost:8508
pytest -q tests
```

## Deploy (Streamlit Community Cloud)

Le modèle `models/modality_clf.joblib` (~220 Ko) est versionné : premier chargement sans entraînement.

1. Repo public : [`EmilienG/microscope-panopticon`](https://github.com/EmilienG/microscope-panopticon)
2. [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Branch `main`
4. **Main file path** : `frontend/app.py`
5. **Python version** : 3.11 ou 3.12
6. Deploy

## Pipeline

1. Générateur de frames par modalité — `src/synth.py`
2. 14 signatures visuelles — `src/features.py`
3. `StandardScaler` + `RandomForest` — `src/classifier.py`
4. UI Streamlit (upload ou galerie synthétique) — `frontend/app.py`

## Classes

| id | UI | Signature visée |
|---|---|---|
| `fluorescence` | Fluorescence | fond noir, émission saturée |
| `brightfield` | Fond clair (H&E) | champ lumineux, rose / violet |
| `phase_contrast` | Contraste de phase | halo, quasi gris |
| `sem` | MEB / SEM | grain, relief, gris |
| `darkfield` | Fond noir | anneaux brillants sur noir |
| `confocal` | Confocal | points nets, fond quasi noir |

## Limites (assumées)

Le classifieur est entraîné sur du **synthétique**. Une photo réelle d’un labo
peut tomber en confiance basse — le PoC montre le *pipeline* (image brute →
modalité), pas un modèle clinique.
