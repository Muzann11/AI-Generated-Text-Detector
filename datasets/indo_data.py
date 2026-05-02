"""
Cek dataset Indonesia yang disarankan
"""
from datasets import load_dataset
from collections import Counter

# ─────────────────────────────────────────────
# 1. M4 — berbagai kemungkinan nama
# ─────────────────────────────────────────────
print("=" * 50)
print("=== M4 Dataset (berbagai nama) ===")
print("=" * 50)
m4_candidates = [
    ("mbzuai-nlp/M4", {}),
    ("mbzuai-nlp/M4", {"name": "indonesian"}),
    ("SemEval2024-Task8", {}),
    ("semeval2024-task8/M4", {}),
    ("ai-detection/M4", {}),
]
for name, kwargs in m4_candidates:
    try:
        ds = load_dataset(name, **kwargs, streaming=True, split="train")
        for row in ds:
            print(f"✅ {name} {kwargs}")
            print(f"   Kolom: {list(row.keys())}")
            for k, v in row.items():
                print(f"   {k}: {str(v)[:80]}")
            break
        break
    except Exception as e:
        print(f"❌ {name} {kwargs}: {e}")

# ─────────────────────────────────────────────
# 2. IndoPref
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("=== IndoPref ===")
print("=" * 50)
indopref_candidates = [
    "IndoPref",
    "indonlp/IndoPref",
    "indopref",
    "LazarusNLP/IndoPref",
]
for name in indopref_candidates:
    try:
        ds = load_dataset(name, streaming=True, split="train")
        for row in ds:
            print(f"✅ {name}")
            print(f"   Kolom: {list(row.keys())}")
            for k, v in row.items():
                print(f"   {k}: {str(v)[:80]}")
            break
        break
    except Exception as e:
        print(f"❌ {name}: {e}")

# ─────────────────────────────────────────────
# 3. Dataset Indo lain yang mungkin berguna
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("=== Kandidat lain ===")
print("=" * 50)
others = [
    "SEACrowd/id_newspapers_2018",
    "id_newspapers_2018",
    "indonlp/indonesian-nlp",
    "wikimedia/wikipedia",
]
for name in others:
    kwargs = {"name": "20231101.id"} if name == "wikimedia/wikipedia" else {}
    try:
        ds = load_dataset(name, **kwargs, streaming=True, split="train")
        for row in ds:
            print(f"✅ {name}")
            print(f"   Kolom: {list(row.keys())}")
            for k, v in row.items():
                print(f"   {k}: {str(v)[:80]}")
            break
    except Exception as e:
        print(f"❌ {name}: {e}")

print("\n" + "=" * 50)
print("Selesai! Paste output ke Claude.")
print("=" * 50)