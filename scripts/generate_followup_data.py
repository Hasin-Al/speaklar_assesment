import json
import random
import re
from pathlib import Path

KB_PATH = Path(__file__).resolve().parent.parent / "Knowledge_Bank.txt"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "followup_pairs.jsonl"

random.seed(7)

context_templates = [
    "{p} আছে কি?",
    "{p} পাওয়া যায়?",
    "{p} সম্পর্কে বলুন",
    "{p} নিয়ে জানতে চাই",
    "{p} স্টকে আছে?",
]

followups = [
    "দাম কত?",
    "ছাড় আছে কি?",
    "ওয়ারেন্টি আছে কি?",
    "ডেলিভারি কেমন?",
    "পেমেন্ট অপশন কী?",
    "রিভিউ কত?",
    "বৈশিষ্ট্য কী?",
    "জনপ্রিয় কি?",
    "স্টকে আছে?",
]

anaphoric = [
    "আর এতে ছাড় আছে?",
    "এইটার রিভিউ কত?",
    "এটার ডেলিভারি আছে?",
    "এইটার ওয়ারেন্টি আছে?",
    "এর দাম কত?",
    "এই পণ্যের বৈশিষ্ট্য কী?",
]

standalone_templates = [
    "{p} এর দাম কত?",
    "{p} তে ছাড় আছে কি?",
    "{p} এর রিভিউ কত?",
    "{p} এর ওয়ারেন্টি আছে কি?",
    "{p} এর ডেলিভারি কেমন?",
    "{p} এর বৈশিষ্ট্য কী?",
]

neg_general = [
    "আজকের আবহাওয়া কেমন?",
    "আপনি কে?",
    "ঢাকায় আজ বৃষ্টি হবে?",
    "তোমার নাম কী?",
]


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

    pairs = []

    # Positive pairs: followup depends on context
    for p in products:
        ctx = random.choice(context_templates).format(p=p)
        f1 = random.choice(followups)
        f2 = random.choice(anaphoric)
        pairs.append({"text_a": f1, "text_b": ctx, "label": 1})
        pairs.append({"text_a": f2, "text_b": ctx, "label": 1})

    # Negative pairs: standalone questions paired with unrelated context
    sample_n = min(len(products), 200)
    for p in random.sample(products, sample_n):
        curr = random.choice(standalone_templates).format(p=p)
        other = random.choice([x for x in products if x != p])
        ctx = random.choice(context_templates).format(p=other)
        pairs.append({"text_a": curr, "text_b": ctx, "label": 0})

    # Negative pairs: general chat with product context
    sample_n2 = min(len(products), 100)
    for p in random.sample(products, sample_n2):
        curr = random.choice(neg_general)
        ctx = random.choice(context_templates).format(p=p)
        pairs.append({"text_a": curr, "text_b": ctx, "label": 0})

    random.shuffle(pairs)

    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"products={len(products)} pairs={len(pairs)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
