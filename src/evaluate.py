# ================================================================
# AI-Generated Text Detector — evaluate.py
# ================================================================
# Jalankan SETELAH training selesai dan model sudah didownload.
# Script ini bisa dijalankan kapan saja secara independen.
#
# Yang dihasilkan:
#   - Confusion matrix (TP, TN, FP, FN) overall & per bahasa
#   - Accuracy, Precision, Recall, F1 per bahasa
#   - Error analysis (contoh teks yang salah diprediksi)
#   - ROC curve & AUC score
#   - Threshold analysis
#   - Semua hasil disimpan ke evaluation_results/
# ================================================================

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
)

# ── Config ───────────────────────────────────────────────────────
MODEL_DIR  = "./model_output/best_model"   # folder hasil download dari Kaggle
DATA_DIR   = "./data"                      # folder data lokal
OUTPUT_DIR = "./evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load train_config.json ───────────────────────────────────────
config_path = "./model_output/train_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        train_cfg = json.load(f)
    MAX_LEN    = train_cfg["max_len"]
    MODEL_NAME = train_cfg["model_name"]
    ID2LABEL   = {int(k): v for k, v in train_cfg["id2label"].items()}
    print(f"Config loaded: model={MODEL_NAME} | max_len={MAX_LEN}")
    print(f"Best val_f1={train_cfg['best_val_f1']:.4f} @ epoch {train_cfg['best_epoch']}")
else:
    # Fallback default
    MAX_LEN    = 256
    MODEL_NAME = "xlm-roberta-large"
    ID2LABEL   = {0: "human", 1: "ai-generated"}
    print("train_config.json tidak ditemukan, pakai default config")

# ── Load Model ───────────────────────────────────────────────────
print(f"\nLoading model dari {MODEL_DIR}...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model = model.to(DEVICE)
model.eval()
print("Model loaded ✅")

# ── Dataset ──────────────────────────────────────────────────────
class EvalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tok    = tokenizer
        self.max_len= max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids"     : enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Predict Function ─────────────────────────────────────────────
def predict(df):
    """Return labels, preds, probs (softmax)"""
    ds     = EvalDataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            logits = out.logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(-1).cpu().numpy()
            all_labels.extend(batch["label"].numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


# ── Helper: Confusion Matrix Breakdown ───────────────────────────
def cm_breakdown(labels, preds):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn),
        "Accuracy" : accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall"   : recall_score(labels, preds, zero_division=0),
        "F1"       : f1_score(labels, preds, average="macro"),
    }


# ── Helper: Plot Confusion Matrix ────────────────────────────────
def plot_confusion_matrix(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Human", "AI"],
                yticklabels=["Human", "AI"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Helper: ROC Curve ────────────────────────────────────────────
def plot_roc(labels, probs, title, save_path):
    fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
    auc = roc_auc_score(labels, probs[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return auc, fpr, tpr, thresholds


# ════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  LOADING TEST DATA")
print("="*55)

test_df = pd.read_csv(f"{DATA_DIR}/final_test.csv")
print(f"Test set: {len(test_df):,} rows")

labels, preds, probs = predict(test_df)


# ── 1. Overall Metrics ───────────────────────────────────────────
print("\n" + "="*55)
print("  1. OVERALL METRICS")
print("="*55)

overall = cm_breakdown(labels, preds)
for k, v in overall.items():
    if isinstance(v, float):
        print(f"  {k:12}: {v:.4f}")
    else:
        print(f"  {k:12}: {v:,}")

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Human", "AI-Generated"]))

# Save overall metrics
pd.DataFrame([overall]).to_csv(f"{OUTPUT_DIR}/overall_metrics.csv", index=False)

# Plot confusion matrix
plot_confusion_matrix(labels, preds,
                      "Overall Confusion Matrix",
                      f"{OUTPUT_DIR}/confusion_matrix_overall.png")


# ── 2. Per-Language Metrics ──────────────────────────────────────
print("\n" + "="*55)
print("  2. PER-LANGUAGE METRICS")
print("="*55)

if "language" in test_df.columns:
    test_df = test_df.copy()
    test_df["pred"]        = preds
    test_df["prob_human"]  = probs[:, 0]
    test_df["prob_ai"]     = probs[:, 1]
    test_df["correct"]     = (test_df["label"] == test_df["pred"]).astype(int)

    lang_results = []
    for lang, grp in test_df.groupby("language"):
        m = cm_breakdown(grp["label"].values, grp["pred"].values)
        m["language"] = lang
        m["n"]        = len(grp)
        lang_results.append(m)

        print(f"\n  [{lang}] n={len(grp):,}")
        print(f"    TP={m['TP']} | TN={m['TN']} | FP={m['FP']} | FN={m['FN']}")
        print(f"    Acc={m['Accuracy']:.4f} | Prec={m['Precision']:.4f} | "
              f"Rec={m['Recall']:.4f} | F1={m['F1']:.4f}")

        # Plot per bahasa
        if len(grp) >= 20:
            plot_confusion_matrix(
                grp["label"].values, grp["pred"].values,
                f"Confusion Matrix [{lang}]",
                f"{OUTPUT_DIR}/confusion_matrix_{lang}.png"
            )

    lang_df = pd.DataFrame(lang_results)
    cols    = ["language", "n", "TP", "TN", "FP", "FN",
               "Accuracy", "Precision", "Recall", "F1"]
    lang_df = lang_df[cols]
    lang_df.to_csv(f"{OUTPUT_DIR}/per_language_metrics.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR}/per_language_metrics.csv")


# ── 3. ROC Curve & AUC ──────────────────────────────────────────
print("\n" + "="*55)
print("  3. ROC CURVE & AUC")
print("="*55)

auc, fpr, tpr, thresholds = plot_roc(
    labels, probs, "ROC Curve (Overall)",
    f"{OUTPUT_DIR}/roc_curve_overall.png"
)
print(f"  Overall AUC: {auc:.4f}")

# Per bahasa
if "language" in test_df.columns:
    for lang, grp in test_df.groupby("language"):
        if len(grp["label"].unique()) < 2:
            continue
        try:
            lang_probs = np.stack([grp["prob_human"].values, grp["prob_ai"].values], axis=1)
            auc_lang, _, _, _ = plot_roc(
                grp["label"].values, lang_probs,
                f"ROC [{lang}]",
                f"{OUTPUT_DIR}/roc_{lang}.png"
            )
            print(f"  [{lang}] AUC: {auc_lang:.4f}")
        except Exception as e:
            print(f"  [{lang}] ROC skip: {e}")


# ── 4. Threshold Analysis ────────────────────────────────────────
print("\n" + "="*55)
print("  4. THRESHOLD ANALYSIS")
print("="*55)
print("  Threshold | Acc    | Prec   | Recall | F1")
print("  " + "-"*50)

thresh_results = []
for thresh in np.arange(0.3, 0.8, 0.05):
    preds_t = (probs[:, 1] >= thresh).astype(int)
    row = {
        "threshold": round(thresh, 2),
        "accuracy" : accuracy_score(labels, preds_t),
        "precision": precision_score(labels, preds_t, zero_division=0),
        "recall"   : recall_score(labels, preds_t, zero_division=0),
        "f1"       : f1_score(labels, preds_t, average="macro"),
    }
    thresh_results.append(row)
    print(f"  {thresh:.2f}      | {row['accuracy']:.4f} | {row['precision']:.4f} | "
          f"{row['recall']:.4f} | {row['f1']:.4f}")

pd.DataFrame(thresh_results).to_csv(f"{OUTPUT_DIR}/threshold_analysis.csv", index=False)


# ── 5. Error Analysis ────────────────────────────────────────────
print("\n" + "="*55)
print("  5. ERROR ANALYSIS")
print("="*55)

test_df["prob_human"] = probs[:, 0]
test_df["prob_ai"]    = probs[:, 1]
test_df["pred"]       = preds
test_df["correct"]    = (test_df["label"] == test_df["pred"]).astype(int)

errors = test_df[test_df["correct"] == 0].copy()
fp_df  = errors[errors["label"] == 0]  # Human diprediksi AI
fn_df  = errors[errors["label"] == 1]  # AI diprediksi Human

print(f"  False Positives (human → AI)  : {len(fp_df):,}")
print(f"  False Negatives (AI → human)  : {len(fn_df):,}")

# Simpan 50 contoh error terbesar (confidence tertinggi tapi salah)
fp_sample = fp_df.nlargest(50, "prob_ai")[["text", "label", "pred", "prob_ai", "language", "source"]]
fn_sample = fn_df.nlargest(50, "prob_human")[["text", "label", "pred", "prob_human", "language", "source"]]

fp_sample.to_csv(f"{OUTPUT_DIR}/error_false_positives.csv", index=False)
fn_sample.to_csv(f"{OUTPUT_DIR}/error_false_negatives.csv", index=False)
print(f"  Saved error samples ke {OUTPUT_DIR}/")


# ── 6. Save Full Predictions ─────────────────────────────────────
test_df.to_csv(f"{OUTPUT_DIR}/all_predictions.csv", index=False)


# ── Summary ──────────────────────────────────────────────────────
print("\n" + "="*55)
print("  EVALUATION SELESAI")
print("="*55)
print(f"\nSemua hasil tersimpan di {OUTPUT_DIR}/:")
for f in sorted(Path(OUTPUT_DIR).glob("*")):
    size = f.stat().st_size / 1024
    print(f"  {f.name} ({size:.0f} KB)")

print(f"\nRingkasan:")
print(f"  Accuracy  : {overall['Accuracy']:.4f}")
print(f"  F1 (macro): {overall['F1']:.4f}")
print(f"  AUC       : {auc:.4f}")
print(f"  TP={overall['TP']} | TN={overall['TN']} | FP={overall['FP']} | FN={overall['FN']}")