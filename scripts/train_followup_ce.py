import json
import os
from pathlib import Path

from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "followup_pairs.jsonl"
MODEL_OUT = Path(__file__).resolve().parent.parent / "models" / "followup-cross-encoder"


def load_data(path: Path):
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            examples.append(InputExample(texts=[item["text_a"], item["text_b"]], label=float(item["label"])))
    return examples


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Data not found: {DATA_PATH}")

    examples = load_data(DATA_PATH)
    if not examples:
        raise SystemExit("No training examples loaded")

    # Multilingual, small, fast cross-encoder
    model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    model = CrossEncoder(model_name, num_labels=1)

    train_loader = DataLoader(examples, shuffle=True, batch_size=16)

    model.fit(
        train_dataloader=train_loader,
        epochs=2,
        warmup_steps=100,
        show_progress_bar=True,
    )

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_OUT))
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
