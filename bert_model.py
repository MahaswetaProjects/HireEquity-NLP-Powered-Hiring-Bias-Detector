"""
bert_model.py
BERT fine-tuning pipeline for hiring bias classification.

This file handles:
  1. Dataset preparation from labeled JDs
  2. Tokenisation with HuggingFace transformers
  3. Fine-tuning BERT for sequence classification
  4. Evaluation with classification report
  5. Model saving for use in analyzer.py

Requirements:
  pip install transformers torch scikit-learn

Usage:
  python bert_model.py

The trained model is saved to ./bert_bias_model/
Load it in analyzer.py: HiringBiasAnalyzer(bert_model_path="./bert_bias_model")
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Try importing torch / transformers ───────────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        AdamW,
        get_linear_schedule_with_warmup,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  torch/transformers not installed. Run: pip install transformers torch")


# ── Dataset ───────────────────────────────────────────────────────────────────

# Labelled JD dataset — 0 = inclusive, 1 = biased
# In production: use a larger dataset (e.g., from Kaggle or manually labelled)
JD_DATASET = [
    # Biased (label=1)
    ("We need a rockstar developer who is a digital native and aggressive problem-solver.", 1),
    ("Looking for a young, energetic salesman who can conquer targets.", 1),
    ("The ideal candidate is a ninja coder from an Ivy League school.", 1),
    ("He should be physically fit and a native English speaker.", 1),
    ("We want a dynamic fresh graduate who is a culture fit for our brotherhood.", 1),
    ("Must be a driven warrior with no employment gaps.", 1),
    ("Looking for a strong, fearless champion to dominate the market.", 1),
    ("The right person will be a tech-savvy digital native who boasts great skills.", 1),
    ("We require a manpower resource who is youthful and hip.", 1),
    ("Perfect English and prestige university background required.", 1),
    ("Seeking an aggressive, competitive guru to lead the team.", 1),
    ("The role requires a decisive, independent businessman mindset.", 1),
    ("We prefer candidates who are up-and-coming with no career gaps.", 1),
    ("Our fast-paced environment needs someone young and outspoken.", 1),
    ("Must demonstrate a dominant personality and conquer challenges.", 1),
    ("We're seeking a warm, nurturing receptionist who is well-spoken.", 1),
    ("Ideal candidate: articulate, cheerful and a culture fit.", 1),
    ("Looking for a fresh graduate who is a natural leader — a real champion.", 1),
    ("Experience: 0-2 years. Must be a digital native and tech-savvy.", 1),
    ("Our rockstar team needs another warrior to join the brotherhood.", 1),

    # Inclusive (label=0)
    ("We are looking for a skilled software engineer with 3+ years of experience.", 0),
    ("The role requires strong analytical thinking and collaborative skills.", 0),
    ("We welcome applicants from all backgrounds and experience levels.", 0),
    ("You will work in a high-output environment with a motivated team.", 0),
    ("Proficiency in Python and SQL is required for this position.", 0),
    ("We value diverse perspectives and encourage everyone to apply.", 0),
    ("The ideal candidate has a track record of delivering results.", 0),
    ("Strong written and verbal communication skills are essential.", 0),
    ("Experience with machine learning frameworks is a plus.", 0),
    ("You will collaborate with cross-functional teams to solve complex problems.", 0),
    ("We are an equal-opportunity employer committed to inclusion.", 0),
    ("Candidates should demonstrate experience with agile methodologies.", 0),
    ("We offer flexible working arrangements and a supportive environment.", 0),
    ("The role involves data analysis, reporting, and stakeholder management.", 0),
    ("We seek someone with strong problem-solving and communication skills.", 0),
    ("Experience in a fast-paced technology environment is advantageous.", 0),
    ("The candidate will manage a team of five and report to the VP.", 0),
    ("We encourage candidates with non-traditional backgrounds to apply.", 0),
    ("Fluency in English is required for this role.", 0),
    ("Responsibilities include leading projects and mentoring junior colleagues.", 0),
]


class JDDataset(Dataset):
    """PyTorch Dataset for job description texts."""

    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens    = True,
            max_length            = self.max_len,
            padding               = "max_length",
            truncation            = True,
            return_attention_mask = True,
            return_tensors        = "pt",
        )
        return {
            "input_ids"     : encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train_bert_classifier(
    dataset      = JD_DATASET,
    model_name   = "bert-base-uncased",
    output_dir   = "./bert_bias_model",
    epochs       = 4,
    batch_size   = 4,
    learning_rate= 2e-5,
    max_len      = 256,
    test_size    = 0.2,
    random_state = 42,
):
    """
    Fine-tune BERT for binary bias classification.
    Saves model and tokenizer to output_dir.
    """
    if not TORCH_AVAILABLE:
        print("❌ Cannot train: torch/transformers not available.")
        return

    print(f"\n{'='*55}")
    print(f"  BERT FINE-TUNING — HIRING BIAS CLASSIFIER")
    print(f"{'='*55}")
    print(f"  Base model    : {model_name}")
    print(f"  Dataset size  : {len(dataset)}")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Learning rate : {learning_rate}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device        : {device}")

    # ── Prepare data ─────────────────────────────────────────────────────────
    texts  = [d[0] for d in dataset]
    labels = [d[1] for d in dataset]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"\n  Train samples : {len(X_train)}")
    print(f"  Test samples  : {len(X_test)}")
    print(f"  Class balance : {sum(y_train)} biased / {len(y_train)-sum(y_train)} inclusive (train)")

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = JDDataset(X_train, y_train, tokenizer, max_len)
    test_dataset  = JDDataset(X_test,  y_test,  tokenizer, max_len)

    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels          = 2,
        output_attentions   = False,
        output_hidden_states= False,
    ).to(device)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(0.1 * total_steps),
        num_training_steps = total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            model.zero_grad()
            outputs = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                labels         = batch["labels"].to(device),
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n  Evaluating on test set...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
            )
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    print("\n  Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Inclusive", "Biased"]
    ))

    cm = confusion_matrix(all_labels, all_preds)
    print(f"  Confusion Matrix:\n{cm}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    meta = {
        "base_model"  : model_name,
        "epochs"      : epochs,
        "dataset_size": len(dataset),
        "labels"      : ["Inclusive", "Biased"],
    }
    with open(os.path.join(output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Model saved to: {output_dir}")
    print(f"{'='*55}\n")
    return model, tokenizer


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_bias_probability(text: str, model_path: str = "./bert_bias_model") -> float:
    """
    Load saved model and return bias probability for a given text.
    Returns float 0–1 (1 = highly biased).
    """
    if not TORCH_AVAILABLE:
        return None

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model     = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=256, padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    return round(probs[0][1].item(), 4)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_bert_classifier()

    print("\nTesting inference on new JD...")
    test_text = "We need a rockstar ninja who is a digital native and culture fit."

    if os.path.exists("./bert_bias_model"):
        prob = predict_bias_probability(test_text)
        print(f"Bias probability: {prob:.2%}")
    else:
        print("Model not found — run training first.")
