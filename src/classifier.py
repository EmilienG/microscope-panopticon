"""MobileNetV2 modality classifier trained on CytoPacq + complementary frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.classes import CLASS_INDEX, CLASSES
from src.dataset import build_dataset, load_real_examples

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "modality_mobilenet.onnx"
METRICS_PATH = ROOT / "data" / "processed" / "metrics.json"
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 → (1, 3, 224, 224) ImageNet-normalized float32."""
    img = Image.fromarray(np.asarray(rgb)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def train(
    per_class: int = 64,
    seed: int = 0,
    model_path: Path | None = None,
    *,
    epochs: int = 6,
    batch_size: int = 32,
    use_cytopacq: bool = True,
    download_cytopacq: bool = False,
) -> dict:
    """Fine-tune ImageNet MobileNetV2, export ONNX."""
    import torch
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    images, labels, sources = build_dataset(
        per_class=per_class,
        seed=seed,
        use_cytopacq=use_cytopacq,
        download_cytopacq=download_cytopacq,
    )
    y = np.array([CLASS_INDEX[lab] for lab in labels], dtype=np.int64)
    x = np.concatenate([preprocess(im) for im in images], axis=0)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, random_state=7, stratify=y
    )
    real_imgs, real_labs = load_real_examples(copies=8, seed=seed + 1)
    if real_imgs:
        x_real = np.concatenate([preprocess(im) for im in real_imgs], axis=0)
        y_real = np.array([CLASS_INDEX[lab] for lab in real_labs], dtype=np.int64)
        x_tr = np.concatenate([x_tr, x_real], axis=0)
        y_tr = np.concatenate([y_tr, y_real], axis=0)
        sources["real_aug"] = int(len(real_imgs))
    else:
        sources["real_aug"] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    net.classifier[1] = nn.Linear(net.last_channel, len(CLASSES))
    net.to(device)

    train_ds = TensorDataset(
        torch.from_numpy(x_tr),
        torch.from_numpy(y_tr),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()

    for p in net.features.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(net.classifier.parameters(), lr=1e-3)
    head_epochs = max(1, epochs // 3)
    net.train()
    for _ in range(head_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if torch.rand(1).item() > 0.5:
                xb = torch.flip(xb, dims=[-1])
            opt.zero_grad(set_to_none=True)
            loss_fn(net(xb), yb).backward()
            opt.step()

    for p in net.features.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    net.train()
    for _ in range(max(1, epochs - head_epochs)):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if torch.rand(1).item() > 0.5:
                xb = torch.flip(xb, dims=[-1])
            opt.zero_grad(set_to_none=True)
            loss_fn(net(xb), yb).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(x_te).to(device)).cpu().numpy()
    pred = logits.argmax(axis=1)
    acc = float(accuracy_score(y_te, pred))
    names = list(CLASSES)
    report = classification_report(
        y_te,
        pred,
        labels=list(range(len(CLASSES))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    path = Path(model_path or MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    net.to("cpu").eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        net,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )

    per_class_f1 = {
        name: round(float(report[name]["f1-score"]), 3)
        for name in CLASSES
        if name in report
    }
    return {
        "accuracy": round(acc, 4),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "per_class_f1": per_class_f1,
        "model": str(path),
        "backbone": "mobilenet_v2_imagenet",
        "dataset": {
            "cytopacq": sources["cytopacq"],
            "synth_fill": sources["synth"],
            "real_aug": sources.get("real_aug", 0),
            "per_class": per_class,
            "simulator": "https://cbia.fi.muni.cz/simulator/index.php",
        },
        "classes": list(CLASSES),
    }


class _OnnxModel:
    def __init__(self, path: Path):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.classes_ = np.array(CLASSES)

    def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
        x = preprocess(rgb)
        logits = self._sess.run(["logits"], {"image": x})[0]
        return _softmax(logits)[0]


def load_model(model_path: Path | None = None) -> _OnnxModel:
    path = Path(model_path or MODEL_PATH)
    if not path.exists():
        train(model_path=path)
    if path.suffix == ".joblib":
        raise ValueError("legacy sklearn joblib is no longer used; expected ONNX")
    return _OnnxModel(path)


def _ensure_onnx(model: object | None) -> _OnnxModel:
    if isinstance(model, _OnnxModel):
        return model
    return load_model()


def predict_proba(rgb: np.ndarray, model: object | None = None) -> dict[str, float]:
    clf = _ensure_onnx(model)
    probs = clf.predict_proba(rgb)
    return {c: float(p) for c, p in zip(CLASSES, probs, strict=True)}


def predict(rgb: np.ndarray, model: object | None = None) -> dict:
    probs = predict_proba(rgb, model=model)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, conf = ranked[0]
    return {
        "label": top,
        "confidence": round(conf, 4),
        "ranking": [{"label": k, "score": round(v, 4)} for k, v in ranked],
        "probs": {k: round(v, 4) for k, v in probs.items()},
    }
