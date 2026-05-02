"""
Dataset Pipeline — AI-Generated Text Detection
================================================
Dataset (verified struktur):
  1. HC3          → jsonl, kolom: human_answers, chatgpt_answers, source
  2. MAGE         → yaful/MAGE, kolom: text, label(0=machine,1=human), src
  3. RAID         → liamdugan/raid, kolom: generation, model, domain
  4. ai-pile      → artem9k/ai-text-detection-pile, kolom: text, source(human/model)

Output: data/train.csv, data/val.csv, data/test.csv
Label  : 0 = human, 1 = AI  (konsisten semua dataset)
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter
import os

OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_PER_SOURCE = 40_000   # cap per dataset agar tidak ada yang dominasi
MIN_LEN        = 50       # karakter minimum
MAX_LEN        = 2048     # karakter maksimum


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def clean_and_cap(df, source_name, max_rows=MAX_PER_SOURCE):
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"]   = df["text"].astype(str).str.strip()
    df["label"]  = df["label"].astype(int)
    df["source"] = source_name
    df = df[df["text"].str.len().between(MIN_LEN, MAX_LEN)]
    df = df.drop_duplicates(subset=["text"])
    if len(df) > max_rows:
        halved = max_rows // 2
        parts  = []
        for lbl, grp in df.groupby("label"):
            parts.append(grp.sample(n=min(halved, len(grp)), random_state=42))
        df = pd.concat(parts)
    n0 = (df["label"] == 0).sum()
    n1 = (df["label"] == 1).sum()
    print(f"    [{source_name}] {len(df):,} rows | human={n0:,} | AI={n1:,}")
    return df[["text", "label", "source"]]


# ─────────────────────────────────────────────
# 1. HC3
# ─────────────────────────────────────────────
def load_hc3():
    print("\n[1/4] HC3...")
    ds = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl",
        split="train"
    )
    records = []
    for row in ds:
        for t in (row.get("human_answers") or []):
            if t and t.strip():
                records.append({"text": t.strip(), "label": 0})
        for t in (row.get("chatgpt_answers") or []):
            if t and t.strip():
                records.append({"text": t.strip(), "label": 1})
    return clean_and_cap(pd.DataFrame(records), "hc3")


# ─────────────────────────────────────────────
# 2. MAGE
# label di MAGE: 0=machine, 1=human → flip ke 0=human, 1=AI
# ─────────────────────────────────────────────
def load_mage():
    print("\n[2/4] MAGE...")
    ds = load_dataset("yaful/MAGE")
    dfs = []
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()[["text", "label"]]
        df["label"] = df["label"].map({0: 1, 1: 0})  # flip
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return clean_and_cap(df, "mage")


# ─────────────────────────────────────────────
# 3. RAID (streaming — sangat besar)
# Ambil hanya attack="none" untuk menghindari teks yang sengaja di-obfuscate
# ─────────────────────────────────────────────
def load_raid():
    print("\n[3/4] RAID (streaming, ini butuh waktu)...")
    ds = load_dataset("liamdugan/raid", streaming=True, split="train")

    # Karena RAID diurutkan per domain+model, human text tersebar di seluruh dataset
    # Kita kumpulkan human dan AI secara terpisah sampai masing-masing cukup
    TARGET_PER_LABEL = MAX_PER_SOURCE // 2  # 20k human + 20k AI
    human_records = []
    ai_records    = []
    MAX_SCAN      = 2_000_000  # batas scan supaya tidak infinite loop
    scanned       = 0

    for row in ds:
        if scanned >= MAX_SCAN:
            print(f"    [raid] Scan limit reached ({MAX_SCAN:,})")
            break
        if len(human_records) >= TARGET_PER_LABEL and len(ai_records) >= TARGET_PER_LABEL:
            break

        scanned += 1
        if row.get("attack", "none") != "none":
            continue

        text  = (row.get("generation") or "").strip()
        model = (row.get("model") or "").lower()

        if not text:
            continue

        if model == "human" and len(human_records) < TARGET_PER_LABEL:
            human_records.append({"text": text, "label": 0})
        elif model != "human" and len(ai_records) < TARGET_PER_LABEL:
            ai_records.append({"text": text, "label": 1})

    print(f"    [raid] Scanned {scanned:,} rows → human={len(human_records):,} | AI={len(ai_records):,}")
    records = human_records + ai_records
    return clean_and_cap(pd.DataFrame(records), "raid")


# ─────────────────────────────────────────────
# 4. ai-text-detection-pile
# source: "human" → 0, nama model → 1
# ─────────────────────────────────────────────
def load_aipile():
    print("\n[4/4] ai-text-detection-pile (streaming)...")
    # Dataset ini diurutkan: semua human dulu, baru AI
    # Kita kumpulkan terpisah supaya dapat keduanya
    ds = load_dataset("artem9k/ai-text-detection-pile", streaming=True, split="train")

    TARGET_PER_LABEL = MAX_PER_SOURCE // 2  # 20k human + 20k AI
    human_records = []
    ai_records    = []
    MAX_SCAN      = 500_000
    scanned       = 0

    for row in ds:
        if scanned >= MAX_SCAN:
            break
        if len(human_records) >= TARGET_PER_LABEL and len(ai_records) >= TARGET_PER_LABEL:
            break

        scanned += 1
        text   = (row.get("text") or "").strip()
        source = (row.get("source") or "").strip()  # pakai nilai asli, jangan lowercase dulu
        label  = 0 if source.lower() == "human" else 1

        if not text:
            continue

        if label == 0 and len(human_records) < TARGET_PER_LABEL:
            human_records.append({"text": text, "label": 0})
        elif label == 1 and len(ai_records) < TARGET_PER_LABEL:
            ai_records.append({"text": text, "label": 1})

    print(f"    [ai_pile] Scanned {scanned:,} rows → human={len(human_records):,} | AI={len(ai_records):,}")
    records = human_records + ai_records
    if not records:
        print("    [ai_pile] WARNING: tidak ada data, skip")
        return pd.DataFrame(columns=["text", "label", "source"])
    return clean_and_cap(pd.DataFrame(records), "ai_pile")


# ─────────────────────────────────────────────
# MERGE + GLOBAL BALANCE
# ─────────────────────────────────────────────
def merge_and_balance(dfs):
    print("\n[Merge] Menggabungkan semua dataset...")
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  Total sebelum balance : {len(merged):,}")
    print(f"  Per source:")
    for src, cnt in merged["source"].value_counts().items():
        print(f"    {src}: {cnt:,}")

    label_counts = Counter(merged["label"])
    print(f"  Label dist sebelum   : human={label_counts[0]:,} | AI={label_counts[1]:,}")

    min_count = min(label_counts.values())
    parts = []
    for lbl, grp in merged.groupby("label"):
        if len(grp) > min_count:
            grp = resample(grp, n_samples=min_count, random_state=42)
        parts.append(grp)
    merged = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    label_counts2 = Counter(merged["label"])
    print(f"  Label dist setelah   : human={label_counts2[0]:,} | AI={label_counts2[1]:,}")
    print(f"  Total final          : {len(merged):,}")
    return merged


# ─────────────────────────────────────────────
# SPLIT (stratified)
# ─────────────────────────────────────────────
def split_and_save(df):
    print("\n[Split] Train 80% / Val 10% / Test 10%...")
    train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val, test   = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

    train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val.to_csv(f"{OUTPUT_DIR}/val.csv",     index=False)
    test.to_csv(f"{OUTPUT_DIR}/test.csv",   index=False)
    df.to_csv(f"{OUTPUT_DIR}/merged_dataset.csv", index=False)

    print(f"  Train : {len(train):,} rows")
    print(f"  Val   : {len(val):,} rows")
    print(f"  Test  : {len(test):,} rows")
    print(f"\n  Semua file tersimpan di {OUTPUT_DIR}/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  AI Text Detection — Dataset Pipeline")
    print("  Label: 0=human | 1=AI")
    print("=" * 55)

    dfs = []
    dfs.append(load_hc3())
    dfs.append(load_mage())
    dfs.append(load_raid())
    dfs.append(load_aipile())

    merged = merge_and_balance(dfs)
    split_and_save(merged)

    print("\n" + "=" * 55)
    print("SELESAI! File siap untuk training.")
    print("=" * 55)
    print("\nBreakdown final per source x label:")
    print(merged.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())
    print("\nNext: jalankan train_model.py")