# ================================================================
# AI-Generated Text Detector — train_kaggle.py
# ================================================================
# Model   : XLM-RoBERTa-large
# Teknik  : LoRA + Weighted Loss + fp16
# Dataset : final_train.csv, final_val.csv, final_test.csv
#
# Setup Kaggle:
#   1. Upload final_train.csv, final_val.csv, final_test.csv
#      ke Kaggle Dataset (New Dataset → upload 3 file)
#   2. Notebook → Add Data → pilih dataset yang diupload
#   3. Aktifkan GPU T4 x2 atau P100
#   4. Settings → Internet → On
#   5. Run semua cell
# ================================================================

# ── Cell 1: Install ──────────────────────────────────────────────
# !pip install -q transformers==4.40.0 peft==0.10.0 accelerate scikit-learn

# ── Cell 2: Imports ──────────────────────────────────────────────
import os, time, json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score, classification_report

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory // 1024**3
        print(f"  GPU {i} : {torch.cuda.get_device_name(i)} ({mem} GB)")

# ── Cell 3: Config ───────────────────────────────────────────────
class CFG:
    # ⚠️  Sesuaikan DATA_DIR dengan nama dataset Kaggle kamu
    DATA_DIR   = "/kaggle/input/ai-text-detection-dataset"
    OUTPUT_DIR = "/kaggle/working/model_output"

    MODEL_NAME  = "xlm-roberta-large"
    MAX_LEN     = 256        # naikkan ke 512 kalau VRAM cukup
    NUM_LABELS  = 2          # 0=human, 1=AI

    # LoRA
    LORA_R       = 16
    LORA_ALPHA   = 32
    LORA_DROPOUT = 0.1

    # Training
    EPOCHS        = 4
    BATCH_SIZE    = 16       # turunkan ke 8 kalau OOM
    GRAD_ACCUM    = 4        # effective batch = 16×4 = 64
    LR            = 2e-4
    WARMUP_RATIO  = 0.1
    MAX_GRAD_NORM = 1.0
    FP16          = True
    SEED          = 42

    # Weighted loss — bobot per bahasa
    LANG_WEIGHTS = {
        "en": 1.0, "id": 3.0,
        "ar": 2.0, "ru": 2.0,
        "bg": 2.0, "de": 2.0,
        "ur": 2.0, "kk": 2.0,
    }
    DEFAULT_WEIGHT = 1.5

    # Label mapping (disimpan untuk inference nanti)
    ID2LABEL = {0: "human", 1: "ai-generated"}
    LABEL2ID = {"human": 0, "ai-generated": 1}

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(CFG.SEED)

# ── Cell 4: Load Data ────────────────────────────────────────────
def load_data():
    train = pd.read_csv(f"{CFG.DATA_DIR}/final_train.csv")
    val   = pd.read_csv(f"{CFG.DATA_DIR}/final_val.csv")
    test  = pd.read_csv(f"{CFG.DATA_DIR}/final_test.csv")
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"{name}: {len(df):,} | human={(df['label']==0).sum():,} | AI={(df['label']==1).sum():,}")
        if "language" in df.columns:
            print(f"  lang: {dict(df['language'].value_counts())}")
    return train, val, test

train_df, val_df, test_df = load_data()

# ── Cell 5: Tokenizer & Dataset ──────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, use_weights=True):
        self.texts    = df["text"].astype(str).tolist()
        self.labels   = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

        if use_weights and "weight" in df.columns:
            self.weights = df["weight"].astype(float).tolist()
        elif use_weights and "language" in df.columns:
            self.weights = [CFG.LANG_WEIGHTS.get(l, CFG.DEFAULT_WEIGHT)
                            for l in df["language"].tolist()]
        else:
            self.weights = [1.0] * len(df)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long),
            "weight"        : torch.tensor(self.weights[idx], dtype=torch.float),
        }

train_ds = TextDataset(train_df, tokenizer, CFG.MAX_LEN, use_weights=True)
val_ds   = TextDataset(val_df,   tokenizer, CFG.MAX_LEN, use_weights=False)
test_ds  = TextDataset(test_df,  tokenizer, CFG.MAX_LEN, use_weights=False)

train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

# ── Cell 6: Model + LoRA ─────────────────────────────────────────
base_model = AutoModelForSequenceClassification.from_pretrained(
    CFG.MODEL_NAME,
    num_labels = CFG.NUM_LABELS,
    id2label   = CFG.ID2LABEL,
    label2id   = CFG.LABEL2ID,
)

lora_cfg = LoraConfig(
    task_type      = TaskType.SEQ_CLS,
    r              = CFG.LORA_R,
    lora_alpha     = CFG.LORA_ALPHA,
    lora_dropout   = CFG.LORA_DROPOUT,
    target_modules = ["query", "key", "value", "dense"],
    bias           = "none",
    modules_to_save= ["classifier"],
)

model  = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"DataParallel: {torch.cuda.device_count()} GPU")
    model = nn.DataParallel(model)

# ── Cell 7: Loss, Optimizer, Scheduler ──────────────────────────
class WeightedCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, labels, weights):
        return (self.ce(logits, labels) * weights).mean()

criterion = WeightedCELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=0.01)

total_steps  = len(train_loader) * CFG.EPOCHS // CFG.GRAD_ACCUM
warmup_steps = int(total_steps * CFG.WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# GradScaler — pakai API baru (tidak deprecated)
scaler = torch.amp.GradScaler("cuda", enabled=CFG.FP16)

print(f"Total steps: {total_steps} | Warmup: {warmup_steps}")

# ── Cell 8: Train & Eval Functions ──────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    optimizer.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(loader):
        ids     = batch["input_ids"].to(device)
        mask    = batch["attention_mask"].to(device)
        labels  = batch["label"].to(device)
        weights = batch["weight"].to(device)

        # API baru autocast
        with torch.amp.autocast("cuda", enabled=CFG.FP16):
            out    = model(input_ids=ids, attention_mask=mask)
            logits = out.logits if hasattr(out, "logits") else out[0]
            loss   = criterion(logits, labels, weights) / CFG.GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % CFG.GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            # Clip hanya trainable params (proper untuk LoRA)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                CFG.MAX_GRAD_NORM
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * CFG.GRAD_ACCUM
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % 200 == 0:
            ela = time.time() - t0
            eta = ela / (step + 1) * (len(loader) - step - 1)
            print(f"  [{time.strftime('%H:%M:%S')}] Ep{epoch} "
                  f"step {step+1}/{len(loader)} | "
                  f"loss={total_loss/(step+1):.4f} | "
                  f"ETA={eta/60:.1f}m")

    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="macro"))


def evaluate(model, loader, split="Val"):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in loader:
            ids     = batch["input_ids"].to(device)
            mask    = batch["attention_mask"].to(device)
            labels  = batch["label"].to(device)
            weights = batch["weight"].to(device)

            with torch.amp.autocast("cuda", enabled=CFG.FP16):
                out    = model(input_ids=ids, attention_mask=mask)
                logits = out.logits if hasattr(out, "logits") else out[0]
                loss   = criterion(logits, labels, weights)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="macro")
    print(f"  [{split}] loss={loss:.4f} | acc={acc:.4f} | f1={f1:.4f}")
    return loss, acc, f1, all_preds, all_labels


# ── Cell 9: Training Loop ────────────────────────────────────────
print("\n" + "="*55)
print(f"  START TRAINING — {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*55)

best_f1, best_epoch, history = 0, 0, []
t0_total = time.time()

for epoch in range(1, CFG.EPOCHS + 1):
    print(f"\n── Epoch {epoch}/{CFG.EPOCHS} ──────────────────────────")

    tr_loss, tr_acc, tr_f1 = train_epoch(
        model, train_loader, optimizer, scheduler, scaler, epoch)
    print(f"  [Train] loss={tr_loss:.4f} | acc={tr_acc:.4f} | f1={tr_f1:.4f}")

    vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, "Val")

    history.append(dict(epoch=epoch,
                        train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                        val_loss=vl_loss,   val_acc=vl_acc,   val_f1=vl_f1))

    if vl_f1 > best_f1:
        best_f1, best_epoch = vl_f1, epoch
        save_path  = f"{CFG.OUTPUT_DIR}/best_model"
        m_to_save  = model.module if hasattr(model, "module") else model
        m_to_save.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  Saved best model (val_f1={vl_f1:.4f}) → {save_path}")

    if epoch - best_epoch >= 2:
        print(f"  Early stopping triggered (best epoch={best_epoch})")
        break

print(f"\nDone. Total: {(time.time()-t0_total)/60:.1f} min | Best val_f1={best_f1:.4f} @ epoch {best_epoch}")

# ── Cell 10: Save semua artefak ──────────────────────────────────
# Training history
pd.DataFrame(history).to_csv(f"{CFG.OUTPUT_DIR}/training_history.csv", index=False)

# Config — disimpan sebagai JSON untuk dipakai saat inference/deploy
config_to_save = {
    "model_name"    : CFG.MODEL_NAME,
    "max_len"       : CFG.MAX_LEN,
    "num_labels"    : CFG.NUM_LABELS,
    "id2label"      : CFG.ID2LABEL,
    "label2id"      : CFG.LABEL2ID,
    "lang_weights"  : CFG.LANG_WEIGHTS,
    "lora_r"        : CFG.LORA_R,
    "lora_alpha"    : CFG.LORA_ALPHA,
    "best_epoch"    : best_epoch,
    "best_val_f1"   : best_f1,
    "trained_at"    : time.strftime('%Y-%m-%d %H:%M:%S'),
}
with open(f"{CFG.OUTPUT_DIR}/train_config.json", "w") as f:
    json.dump(config_to_save, f, indent=2)

print(f"\nArtefak tersimpan di {CFG.OUTPUT_DIR}/:")
for f in sorted(Path(CFG.OUTPUT_DIR).rglob("*")):
    if f.is_file():
        size = f.stat().st_size / 1024**2
        print(f"  {f.relative_to(CFG.OUTPUT_DIR)} ({size:.1f} MB)")

# ── Cell 11: Quick test set evaluation ──────────────────────────
print("\n" + "="*55)
print("  TEST SET EVALUATION")
print("="*55)

_, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, "Test")

print("\nClassification Report (overall):")
print(classification_report(test_labels, test_preds,
                             target_names=["Human", "AI-Generated"]))

# Per-bahasa
if "language" in test_df.columns:
    print("Per-Language:")
    test_df = test_df.copy()
    test_df["pred"] = test_preds
    for lang, grp in test_df.groupby("language"):
        a = accuracy_score(grp["label"], grp["pred"])
        f = f1_score(grp["label"], grp["pred"], average="macro")
        print(f"  [{lang}] n={len(grp):,} | acc={a:.4f} | f1={f:.4f}")

# Simpan prediksi test set lengkap untuk analisis offline
test_df["pred"]       = test_preds
test_df["correct"]    = (test_df["label"] == test_df["pred"]).astype(int)
test_df.to_csv(f"{CFG.OUTPUT_DIR}/test_predictions.csv", index=False)
print(f"\nPrediksi test set → {CFG.OUTPUT_DIR}/test_predictions.csv")
print("\nTraining selesai! Download semua file di OUTPUT_DIR.")
print(f"   Path: {CFG.OUTPUT_DIR}")