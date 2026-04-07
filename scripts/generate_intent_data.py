import json
import random
import re
from pathlib import Path

KB_PATH = Path(__file__).resolve().parent.parent / "Knowledge_Bank.txt"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "intent_train.jsonl"

random.seed(7)

INTENT_TEMPLATES = {
    "availability": [
        "{p} আছে কি?",
        "{p} পাওয়া যায়?",
        "{p} স্টকে আছে?",
        "{p} পাওয়া যাবে?",
        "{p} মজুদ আছে কি?",
    ],
    "price": [
        "{p} এর দাম কত?",
        "{p} এর মূল্য কত?",
        "{p} দাম জানাবেন?",
        "{p} এর প্রাইস কত?",
    ],
    "discount": [
        "{p} তে ছাড় আছে কি?",
        "{p} এ ডিসকাউন্ট আছে?",
        "{p} এর অফার কী?",
        "{p} এ বিশেষ মূল্য আছে?",
    ],
    "warranty": [
        "{p} এর ওয়ারেন্টি আছে কি?",
        "{p} এর warranty আছে?",
        "{p} এর গ্যারান্টি আছে কি?",
    ],
    "delivery": [
        "{p} এর ডেলিভারি কেমন?",
        "{p} এর হোম ডেলিভারি আছে?",
        "{p} দ্রুত ডেলিভারি আছে?",
    ],
    "payment": [
        "{p} এর পেমেন্ট অপশন কী?",
        "{p} পেমেন্ট কীভাবে করব?",
        "{p} এর payment অপশন কী?",
    ],
    "rating": [
        "{p} এর রিভিউ কত?",
        "{p} এর রেটিং কত?",
        "{p} কত স্টার রেটিং?",
    ],
    "features": [
        "{p} এর বৈশিষ্ট্য কী?",
        "{p} এর ফিচার কী?",
        "{p} সম্পর্কে বলুন",
    ],
    "popularity": [
        "{p} কি জনপ্রিয়?",
        "{p} কি বেশি বিক্রি হয়?",
        "{p} কি বহুল ব্যবহৃত?",
    ],
}

GENERAL_INTENT = [
    "আপনি কে?",
    "আজকের আবহাওয়া কেমন?",
    "তোমার নাম কী?",
    "ঢাকায় আজ বৃষ্টি হবে?",
    "সাধারণ প্রশ্ন আছে",
]

GREETING_INTENT = [
    "হ্যালো",
    "হাই",
    "আসসালামু আলাইকুম",
    "কেমন আছেন?",
]

GENERIC_INTENT = {
    "availability": [
        "আছে কি?",
        "স্টকে আছে?",
        "পাওয়া যাবে?",
    ],
    "price": [
        "দাম কত?",
        "মূল্য কত?",
        "প্রাইস কত?",
    ],
    "discount": [
        "ছাড় আছে কি?",
        "ডিসকাউন্ট আছে?",
        "অফার আছে?",
    ],
    "warranty": [
        "ওয়ারেন্টি আছে?",
        "গ্যারান্টি আছে?",
    ],
    "delivery": [
        "ডেলিভারি কেমন?",
        "হোম ডেলিভারি আছে?",
    ],
    "payment": [
        "পেমেন্ট অপশন কী?",
        "payment কীভাবে করব?",
    ],
    "rating": [
        "রিভিউ কত?",
        "রেটিং কত?",
    ],
    "features": [
        "বৈশিষ্ট্য কী?",
        "ফিচার কী?",
    ],
    "popularity": [
        "জনপ্রিয় কি?",
        "বেশি বিক্রি হয়?",
    ],
}


def extract_products(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    names = []
    for p in paras:
        first = p.split("।", 1)[0].strip()
        if first:
            names.append(first)
    seen = set()
    products = []
    for n in names:
        if n not in seen:
            seen.add(n)
            products.append(n)
    return products


def main() -> None:
    text = KB_PATH.read_text(encoding="utf-8")
    products = extract_products(text)

    items = []

    for p in products:
        for intent, templates in INTENT_TEMPLATES.items():
            for t in templates:
                items.append({"text": t.format(p=p), "label": intent})

    for intent, samples in GENERIC_INTENT.items():
        for t in samples:
            items.append({"text": t, "label": intent})

    for t in GENERAL_INTENT:
        items.append({"text": t, "label": "general"})

    for t in GREETING_INTENT:
        items.append({"text": t, "label": "greeting"})

    random.shuffle(items)

    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"products={len(products)} samples={len(items)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
