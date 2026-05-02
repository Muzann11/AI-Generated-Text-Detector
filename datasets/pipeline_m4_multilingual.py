"""
Dataset Pipeline — M4 Multilingual + Indonesian Expansion
===========================================================
Sumber:
  1. M4 (semua bahasa dari GitHub clone)     → human_text vs machine_text
  2. id_newspapers_2018 + Qwen3-32b          → expand domain Indo

Struktur file M4: {"human_text": "...", "machine_text": "...", "model": "...", "source": "..."}

Output: data/m4_multilingual.csv  → siap digabung dengan pipeline English
Label : 0 = human, 1 = AI
"""

import os
import json
import glob
import time
import pandas as pd
import requests
from collections import Counter

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
M4_DIR      = "./M4/data"           # folder hasil git clone
OUTPUT_DIR  = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Proporsi per bahasa — Indo lebih banyak karena prioritas
LANG_CAPS = {
    "id" : 6_000,   # Indonesian — ambil semua (~6k)
    "en" : 5_000,   # English M4 (arxiv, reddit, wikihow, wikipedia, peerread)
    "ar" : 3_000,   # Arabic
    "ru" : 3_000,   # Russian
    "bg" : 3_000,   # Bulgarian
    "de" : 3_000,   # German
    "ur" : 3_000,   # Urdu
    "kk" : 3_000,   # Kazakh
}

# Mapping prefix filename → kode bahasa
FILE_LANG_MAP = {
    "id-newspaper"                  : "id",
    "arabic"                        : "ar",
    "russian"                       : "ru",
    "bulgarian"                     : "bg",
    "germanwikipedia"               : "de",
    "urdu"                          : "ur",
    "qazh"                          : "kk",
    "arxiv"                         : "en",
    "reddit"                        : "en",
    "wikihow"                       : "en",
    "wikipedia"                     : "en",
    "peerread"                      : "en",
}


# ─────────────────────────────────────────────
# 1. LOAD SEMUA FILE M4
# ─────────────────────────────────────────────
def load_m4_all():
    print("\n[1] Loading semua file M4...")
    jsonl_files = glob.glob(os.path.join(M4_DIR, "*.jsonl"))
    print(f"    Ditemukan {len(jsonl_files)} file")

    records_per_lang = {lang: [] for lang in LANG_CAPS}

    for filepath in sorted(jsonl_files):
        filename = os.path.basename(filepath).lower()

        # Tentukan bahasa dari nama file
        lang = None
        for prefix, code in FILE_LANG_MAP.items():
            if filename.startswith(prefix):
                lang = code
                break
        if lang is None:
            print(f"    [skip] tidak dikenali: {filename}")
            continue

        cap = LANG_CAPS.get(lang, 0)
        if not cap:
            continue

        # Parse jsonl
        file_records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    human_raw   = row.get("human_text") or ""
                    machine_raw = row.get("machine_text") or ""
                    human   = " ".join(human_raw).strip()   if isinstance(human_raw,   list) else str(human_raw).strip()
                    machine = " ".join(machine_raw).strip() if isinstance(machine_raw, list) else str(machine_raw).strip()
                    model = row.get("model", "unknown")
                    source = row.get("source", "m4")
                    if human:
                        file_records.append({
                            "text": human, "label": 0,
                            "language": lang, "source": f"m4_{lang}",
                            "model_generator": "human"
                        })
                    if machine:
                        file_records.append({
                            "text": machine, "label": 1,
                            "language": lang, "source": f"m4_{lang}",
                            "model_generator": model
                        })
                except json.JSONDecodeError:
                    continue

        records_per_lang[lang].extend(file_records)
        print(f"    {filename}: {len(file_records)//2} pasang → lang={lang}")

    # Cap per bahasa dan balance
    all_records = []
    for lang, records in records_per_lang.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        cap = LANG_CAPS[lang]
        half = cap // 2
        parts = []
        for lbl, grp in df.groupby("label"):
            parts.append(grp.sample(n=min(half, len(grp)), random_state=42))
        df = pd.concat(parts)
        n0 = (df["label"] == 0).sum()
        n1 = (df["label"] == 1).sum()
        print(f"    [{lang}] {len(df):,} rows | human={n0:,} | AI={n1:,}")
        all_records.append(df)

    return pd.concat(all_records, ignore_index=True)


# ─────────────────────────────────────────────
# 2. GENERATE AI TEXT INDO PAKAI QWEN3
#    (expand domain selain berita)
# ─────────────────────────────────────────────
def generate_indo_qwen3(n_samples=2000, api_url=None, api_key=None, model_name="qwen3-32b"):
    """
    Ambil human text dari Wikipedia Indo (berbagai topik), generate versi AI pakai Qwen3.
    n_samples: jumlah pasang yang ingin digenerate
    """
    import random, time

    if api_url is None:
        print("\n[2] Skip generate Qwen3 — api_url tidak diset")
        return pd.DataFrame()

    print(f"\n[2] Generate Indo via Qwen3 ({n_samples} sampel)...")

    # Ambil human text dari Wikipedia Indo — skip N_SKIP artikel pertama
    # supaya dapat topik yang bervariasi, bukan hanya artikel pertama
    from datasets import load_dataset
    wiki_id = load_dataset("wikimedia/wikipedia", "20231101.id", streaming=True, split="train")

    human_texts  = []
    seen_titles  = set()
    N_SKIP       = 0   # mulai dari awal, tapi filter duplikat judul
    SAMPLE_POOL  = n_samples * 3  # ambil pool lebih besar lalu shuffle

    for row in wiki_id:
        if len(human_texts) >= SAMPLE_POOL:
            break
        title = (row.get("title") or "").strip()
        text  = (row.get("text")  or "").strip()

        if title in seen_titles:
            continue
        seen_titles.add(title)

        # Ambil paragraf pertama saja (lebih natural daripada potong mentah)
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 150]
        if not paragraphs:
            continue

        human_texts.append({
            "title": title,
            "text" : paragraphs[0][:600]
        })

    # Shuffle supaya urutan topik random
    random.shuffle(human_texts)
    human_texts = human_texts[:n_samples]
    print(f"    Pool {len(human_texts)} topik unik dari Wikipedia Indo")
    print(f"    Contoh topik: {[h['title'] for h in human_texts[:5]]}")

    # Template prompt yang bervariasi
    prompt_templates = [
        "Tulis sebuah paragraf informatif dalam bahasa Indonesia tentang topik berikut. Gunakan gaya bahasamu sendiri.\nTopik: {title}\n\nKonteks: {text}",
        "Jelaskan secara singkat dalam bahasa Indonesia mengenai '{title}'. Buat dalam 2-3 paragraf.\n\nInfo dasar: {text}",
        "Buatkan artikel pendek bahasa Indonesia tentang: {title}\n\nReferensi: {text}",
        "Tuliskan sebuah paragraf dalam bahasa Indonesia yang membahas topik ini: {title}",
        "Sebagai penulis, jelaskan tentang '{title}' dalam bahasa Indonesia yang mudah dipahami.\n\nLatar belakang: {text}",
    ]

    # Generate AI text
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    records = []
    failed  = 0

    t_start = time.time()

    for i, item in enumerate(human_texts):
        if i % 50 == 0 or i == 0:
            elapsed   = time.time() - t_start
            remaining = (elapsed / max(i, 1)) * (len(human_texts) - i) if i > 0 else 0
            print(f"    [{time.strftime('%H:%M:%S')}] Progress: {i}/{len(human_texts)} | "
                  f"success={len(records)//2} | failed={failed} | "
                  f"elapsed={elapsed:.0f}s | ETA={remaining:.0f}s")

        # Pilih template secara random
        template = random.choice(prompt_templates)
        prompt   = template.format(title=item["title"], text=item["text"][:300])

        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. /no_think"},
                        {"role": "user",   "content": prompt}
                    ],
                    "max_tokens": 400,
                    "temperature": 0.7,
                },
                timeout=60
            )
            resp.raise_for_status()
            ai_text = resp.json()["choices"][0]["message"]["content"].strip()

            # Filter output thinking tags kalau ada
            if "<think>" in ai_text:
                ai_text = ai_text.split("</think>")[-1].strip()

            if len(ai_text) > 100:
                records.append({"text": item["text"], "label": 0,
                                 "language": "id", "source": "wiki_id_human",
                                 "model_generator": "human"})
                records.append({"text": ai_text, "label": 1,
                                 "language": "id", "source": "wiki_id_qwen3",
                                 "model_generator": model_name})
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"    [warn] Failed row {i} ({item['title']}): {e}")
            continue

    t_total = time.time() - t_start
    df = pd.DataFrame(records)
    print(f"    [qwen3_indo] {len(df):,} rows | success={len(df)//2} | failed={failed} | "
          f"total_time={t_total/60:.1f} menit ({t_total:.0f}s) | "
          f"avg={t_total/max(len(human_texts),1):.1f}s/req)")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3-url",  default=None,  help="Base URL vLLM endpoint, e.g. http://192.168.1.10:8000")
    parser.add_argument("--qwen3-key",  default=None,  help="API key kalau ada (opsional)")
    parser.add_argument("--qwen3-model",default="qwen3-32b", help="Nama model di endpoint")
    parser.add_argument("--n-generate", default=2000, type=int, help="Jumlah sampel Indo yang digenerate")
    args = parser.parse_args()

    t_main = time.time()
    print("=" * 55)
    print("  M4 Multilingual + Indo Expansion Pipeline")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Load M4
    df_m4 = load_m4_all()
    print(f"\n  M4 total: {len(df_m4):,} rows")
    print(f"  Bahasa  : {dict(df_m4['language'].value_counts())}")

    # Generate Indo (opsional)
    df_qwen = generate_indo_qwen3(
        n_samples   = args.n_generate,
        api_url     = args.qwen3_url,
        api_key     = args.qwen3_key,
        model_name  = args.qwen3_model,
    )

    # Gabung
    dfs = [df_m4]
    if not df_qwen.empty:
        dfs.append(df_qwen)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Simpan
    out_path = f"{OUTPUT_DIR}/m4_multilingual.csv"
    merged.to_csv(out_path, index=False)

    print("\n" + "=" * 55)
    print("SELESAI!")
    print("=" * 55)
    print(f"  Output : {out_path}")
    print(f"  Total  : {len(merged):,} rows")
    print(f"\n  Breakdown per bahasa x label:")
    print(merged.groupby(["language", "label"]).size().unstack(fill_value=0).to_string())
    print("\nNext: gabungkan dengan data English di merge_all.py")