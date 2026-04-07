import json
from pathlib import Path

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "intent_train.jsonl"
MODEL_OUT = Path(__file__).resolve().parent.parent / "models" / "intent_clf.npz"
DENSE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

if os.getenv("INTENT_TRAIN_ONLINE", "0") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resolve_local_model_path(model_name: str) -> str:
    repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    snaps = repo_dir / "snapshots"
    if snaps.exists():
        candidates = sorted([p for p in snaps.iterdir() if p.is_dir()])
        if candidates:
            return str(candidates[-1])
    return model_name


def load_data(path: Path):
    texts = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])
    return texts, labels


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Data not found: {DATA_PATH}")

    texts, labels = load_data(DATA_PATH)
    if not texts:
        raise SystemExit("No training data")

    label_list = sorted(set(labels))
    label_index = {l: i for i, l in enumerate(label_list)}
    y = np.array([label_index[l] for l in labels], dtype=np.int64)

    local_only = os.getenv("INTENT_TRAIN_ONLINE", "0") != "1"
    model_path = _resolve_local_model_path(DENSE_MODEL)
    model = SentenceTransformer(model_path, local_files_only=local_only)
    X = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    X = X.astype(np.float32)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    clf = nn.Linear(X_t.shape[1], len(label_list))
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    clf.train()
    for _ in range(4):
        for xb, yb in loader:
            opt.zero_grad()
            logits = clf(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    W = clf.weight.detach().cpu().numpy().astype(np.float32)
    b = clf.bias.detach().cpu().numpy().astype(np.float32)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_OUT, W=W, b=b, labels=np.array(label_list, dtype=object))
    print(f"Saved intent classifier to {MODEL_OUT}")


if __name__ == "__main__":
    main()
