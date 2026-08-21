"""Target modalities: identify the microscope type from a raw frame."""

from __future__ import annotations

CLASSES: tuple[str, ...] = (
    "fluorescence",
    "brightfield",
    "phase_contrast",
    "sem",
    "darkfield",
    "confocal",
)

LABELS_FR: dict[str, str] = {
    "fluorescence": "Fluorescence",
    "brightfield": "Fond clair (H&E)",
    "phase_contrast": "Contraste de phase",
    "sem": "MEB / SEM",
    "darkfield": "Fond noir",
    "confocal": "Confocal",
}

BLURBS: dict[str, str] = {
    "fluorescence": "Fond noir, émission saturée (GFP / DAPI / Texas Red).",
    "brightfield": "Champ lumineux, tissus rose-violet (hématoxyline-éosine).",
    "phase_contrast": "Halo clair autour de cellules grises, presque monochrome.",
    "sem": "Relief métallique, grain élevé, niveaux de gris.",
    "darkfield": "Silhouettes brillantes sur noir, très peu de chroma.",
    "confocal": "Points fluorescents nets, fond quasi noir, optique sectionnée.",
}

CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
