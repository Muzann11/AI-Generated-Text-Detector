"""
merge_all.py — Gabungkan semua dataset jadi satu file final
============================================================
Input:
  - data/merged_dataset.csv     (English: HC3 + MAGE + RAID)
  - data/m4_multilingual.csv    (M4 semua bahasa + optional Qwen3 Indo)

Output:
  - data/final_train.csv
  - data/final_val.csv
  - data/final_test.csv
  - data/final_merged.csv       (semua sebelum split)

Kolom output: text, label, language, source, weight
  label    : 0=human, 1=AI
  language : en, id, ar, ru, bg, de, ur, kk
  weight   : bobot untuk weighted loss saat training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os

OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# BOBOT PER BAHASA (untuk weighted loss)
# Indo prioritas tinggi, English normal
# ─────────────────────────────────────────────
LANG_WEIGHTS = {
    "en" : 1.0,
    "id" : 3.0,
    "ar" : 2.0,
    "ru" : 2.0,
    "bg" : 2.0,
    "de" : 2.0,
    "ur" : 2.0,
    "kk" : 2.0,
}
DEFAULT_WEIGHT = 1.5  # untuk bahasa yang tidak ada di mapping


# ─────────────────────────────────────────────
# 1. LOAD ENGLISH DATA
# ─────────────────────────────────────────────
def load_english():
    path = f"{OUTPUT_DIR}/merged_dataset.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}\nJalankan dataset_pipeline.py dulu!")
    df = pd.read_csv(path)
    # Kolom yang ada: text, label, source
    # Tambah kolom language = "en"
    df["language"] = "en"
    print(f"  [English] {len(df):,} rows | "
          f"human={( df['label']==0).sum():,} | AI={(df['label']==1).sum():,}")
    return df[["text", "label", "language", "source"]]


# ─────────────────────────────────────────────
# 2. LOAD M4 MULTILINGUAL
# ─────────────────────────────────────────────
def load_multilingual():
    path = f"{OUTPUT_DIR}/m4_multilingual.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}\nJalankan pipeline_m4_multilingual.py dulu!")
    df = pd.read_csv(path)

    # Pastikan kolom language ada
    if "language" not in df.columns:
        df["language"] = "unknown"

    print(f"  [M4 Multilingual] {len(df):,} rows")
    for lang, count in df["language"].value_counts().items():
        n0 = ((df["language"] == lang) & (df["label"] == 0)).sum()
        n1 = ((df["language"] == lang) & (df["label"] == 1)).sum()
        print(f"    [{lang}] {count:,} | human={n0:,} | AI={n1:,}")

    # Pastikan ada kolom source
    if "source" not in df.columns:
        df["source"] = "m4"

    return df[["text", "label", "language", "source"]]


# ─────────────────────────────────────────────
# 3. MERGE & ASSIGN WEIGHTS
# ─────────────────────────────────────────────
def merge_and_assign_weights(df_en, df_multi):
    print("\n[Merge] Menggabungkan semua dataset...")

    merged = pd.concat([df_en, df_multi], ignore_index=True)

    # Cleaning
    merged = merged.dropna(subset=["text", "label"])
    merged["text"]  = merged["text"].astype(str).str.strip()
    merged["label"] = merged["label"].astype(int)
    merged = merged[merged["text"].str.len() >= 50]
    merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Assign weight per bahasa
    merged["weight"] = merged["language"].map(LANG_WEIGHTS).fillna(DEFAULT_WEIGHT)

    print(f"\n  Total rows    : {len(merged):,}")
    print(f"  Label dist    : human={(merged['label']==0).sum():,} | AI={(merged['label']==1).sum():,}")
    print(f"\n  Per bahasa:")
    for lang, grp in merged.groupby("language"):
        w = LANG_WEIGHTS.get(lang, DEFAULT_WEIGHT)
        print(f"    [{lang}] {len(grp):,} rows | weight={w} | "
              f"human={(grp['label']==0).sum():,} | AI={(grp['label']==1).sum():,}")

    return merged


# ─────────────────────────────────────────────
# 4. SPLIT — stratified per bahasa + label
# ─────────────────────────────────────────────
def split_stratified(df, train_ratio=0.8, val_ratio=0.1):
    print("\n[Split] Train 80% / Val 10% / Test 10% (stratified per bahasa)...")

    train_parts, val_parts, test_parts = [], [], []

    for lang, group in df.groupby("language"):
        # Perlu minimal 10 rows per label per bahasa untuk split
        for lbl, subgrp in group.groupby("label"):
            if len(subgrp) < 10:
                train_parts.append(subgrp)
                continue

            train, temp = train_test_split(
                subgrp, test_size=(1 - train_ratio),
                random_state=42
            )
            val_size = val_ratio / (1 - train_ratio)
            val, test = train_test_split(
                temp, test_size=(1 - val_size),
                random_state=42
            )
            train_parts.append(train)
            val_parts.append(val)
            test_parts.append(test)

    train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df   = pd.concat(val_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Train : {len(train_df):,} rows")
    print(f"  Val   : {len(val_df):,} rows")
    print(f"  Test  : {len(test_df):,} rows")

    # Cek representasi Indo di semua split
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        indo = (split["language"] == "id").sum()
        print(f"  {name} — Indo: {indo:,} ({indo/len(split)*100:.1f}%)")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Merge All Datasets — Final Pipeline")
    print("=" * 55)

    print("\n[Load] Membaca dataset...")
    df_en    = load_english()
    df_multi = load_multilingual()

    merged = merge_and_assign_weights(df_en, df_multi)

    train_df, val_df, test_df = split_stratified(merged)

    # Simpan
    merged.to_csv(f"{OUTPUT_DIR}/final_merged.csv",  index=False)
    train_df.to_csv(f"{OUTPUT_DIR}/final_train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_DIR}/final_val.csv",     index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/final_test.csv",   index=False)

    print(f"\n  Saved ke {OUTPUT_DIR}/")
    print("  - final_merged.csv")
    print("  - final_train.csv")
    print("  - final_val.csv")
    print("  - final_test.csv")

    print("\n" + "=" * 55)
    print("SELESAI! Data siap untuk training di Kaggle.")
    print("=" * 55)
    print("\nKolom output:")
    print(f"  {list(merged.columns)}")
    print("\nContoh row:")
    print(merged.sample(3)[["text", "label", "language", "weight"]].to_string())
    print("\nNext: upload final_train.csv, final_val.csv, final_test.csv ke Kaggle Dataset")
    print("      lalu jalankan train_model.py di Kaggle Notebook")