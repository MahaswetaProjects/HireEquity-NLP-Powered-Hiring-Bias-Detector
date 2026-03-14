# Hiring Bias Detector

> **NLP + BERT system to detect and fix gender, age, ability & cultural bias in job descriptions**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-BERT-yellow?logo=huggingface)](https://huggingface.co)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi)](https://powerbi.microsoft.com)

---

## The Problem

Biased language in job descriptions silently reduces applicant diversity — before a single interview happens.

| Research Finding | Impact |
|---|---|
| Masculine-coded JDs | 42% fewer women apply (Gaucher et al., 2011) |
| Age-biased JDs | Deters qualified candidates over 40 |
| 'Culture fit' language | Most common proxy for unconscious bias |
| Diverse teams | Outperform homogeneous teams by 35% (McKinsey, 2023) |

---

## What This Tool Does

1. **Scans** any job description using a curated lexicon of 80+ biased phrases
2. **Scores** bias severity per category (gender / age / ability / culture) on a 0–100 scale
3. **Suggests** neutral alternatives for every flagged word
4. **Classifies** the overall JD using fine-tuned BERT
5. **Rewrites** the JD with biased phrases replaced automatically
6. **Exports** trend data ready for Power BI dashboards

---

## Bias Categories Detected

| Category | Example Biased Phrase | Neutral Alternative |
|---|---|---|
| **Gender** | "rockstar", "ninja", "aggressive", "he/him" | "expert", "specialist", "assertive", "they/them" |
| **Age** | "digital native", "young", "fresh graduate" | "tech-proficient", "motivated", "entry-level" |
| **Ability** | "physically fit", "native speaker", "no gaps" | "meets role requirements", "fluent in English" |
| **Culture** | "culture fit", "Ivy League", "top-tier school" | "values alignment", "accredited university" |

---

## Repository Structure

```
📦 hiring-bias-detector/
├── 📓 Hiring_Bias_Detector.ipynb  ← Full analysis notebook
├── 🖥️  app.py                      ← Streamlit interactive dashboard
├── 🧠 analyzer.py                 ← Core bias analysis engine
├── 📖 bias_lexicon.py             ← 80+ phrase bias dictionary
├── 🤖 bert_model.py              ← BERT fine-tuning pipeline
├── 📊 generate_dataset.py        ← Synthetic JD dataset generator
├── 📋 POWERBI_SETUP.md           ← Power BI dashboard guide
├── 📄 requirements.txt           ← Dependencies
└── 📄 README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit dashboard
streamlit run app.py

# 3. Or run the notebook
jupyter notebook Hiring_Bias_Detector.ipynb

# 4. Train the BERT model (optional, requires GPU)
python bert_model.py
```

---

## Streamlit Dashboard Features

| Tab | Features |
|---|---|
| **Analyze JD** | Paste any JD → instant bias score, flagged words, rewritten version |
| **Batch Analysis** | Upload CSV of JDs → analyze all at once |
| **Bias Lexicon** | Browse all 80+ biased phrases with filters |

---

## BERT Model

The `bert_model.py` fine-tunes `bert-base-uncased` for binary bias classification:

```
Architecture : BERT-base-uncased (12 layers, 110M parameters)
Task         : Sequence Classification (2 labels: Inclusive / Biased)
Optimizer    : AdamW (lr=2e-5)
Epochs       : 4
Expected F1  : ~85%
```

---

## Scoring Logic

```
Bias Score (0–100) = Σ(severity_weight × match) / word_count × 100

Severity weights:
  Low (1)      → 5 points
  Moderate (2) → 12 points
  High (3)     → 25 points

Verdict thresholds:
  0–10   → Inclusive       ✅
  10–25  → Mildly Biased   ⚠️
  25–50  → Biased          🚨
  50+    → Highly Biased   🔴
```

---

## Power BI Dashboard

See `POWERBI_SETUP.md` for full setup instructions. Key visuals:
- Bias score distribution by industry and role
- Category heatmap (role × gender/age/ability/culture)
- KPI cards: % inclusive JDs, avg bias score
- Trend line across JD submissions

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core analysis |
| **BERT (HuggingFace Transformers)** | Bias classification model |
| **Statsmodels / scikit-learn** | Evaluation metrics |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Visualisations |
| **Streamlit** | Interactive dashboard |
| **Power BI** | Executive trends dashboard |

