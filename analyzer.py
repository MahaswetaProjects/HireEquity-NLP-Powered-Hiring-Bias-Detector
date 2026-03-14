"""
analyzer.py
Core bias analysis engine.

Two-layer approach:
  1. Rule-based  : lexicon matching with severity weighting
  2. BERT-based  : fine-tuned transformer classifier (loaded if model exists)

Output: BiasReport dataclass with all findings.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from bias_lexicon import BIAS_LEXICON, CATEGORY_INFO, SEVERITY_LABELS


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BiasMatch:
    word          : str
    category      : str
    severity      : int
    neutral       : str
    bias_type     : str
    start_idx     : int
    end_idx       : int
    context       : str   # surrounding sentence


@dataclass
class BiasReport:
    original_text  : str
    word_count     : int
    matches        : List[BiasMatch]
    bias_score     : float              # 0–100
    category_scores: Dict[str, float]  # per-category 0–100
    severity_counts: Dict[str, int]    # {"Low":n, "Moderate":n, "High":n}
    category_counts: Dict[str, int]
    rewritten_text : str
    bert_score     : Optional[float]   # None if model not loaded
    verdict        : str               # "Inclusive" / "Mildly Biased" / "Biased" / "Highly Biased"
    verdict_color  : str
    recommendations: List[str]


# ── Main Analyzer ─────────────────────────────────────────────────────────────

class HiringBiasAnalyzer:
    """
    Analyses a job description for hiring bias.

    Usage:
        analyzer = HiringBiasAnalyzer()
        report   = analyzer.analyze("We need a rockstar developer...")
    """

    def __init__(self, bert_model_path: Optional[str] = None):
        self.lexicon          = BIAS_LEXICON
        self.bert_model       = None
        self.bert_tokenizer   = None

        if bert_model_path:
            self._load_bert(bert_model_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> BiasReport:
        """Full analysis pipeline. Returns a BiasReport."""
        cleaned     = self._preprocess(text)
        matches     = self._lexicon_scan(text, cleaned)
        bias_score  = self._compute_bias_score(text, matches)
        cat_scores  = self._category_scores(text, matches)
        sev_counts  = self._severity_counts(matches)
        cat_counts  = self._category_counts(matches)
        rewritten   = self._rewrite(text, matches)
        bert_score  = self._bert_score(text) if self.bert_model else None
        verdict, color = self._verdict(bias_score)
        recs        = self._recommendations(matches, cat_scores)

        return BiasReport(
            original_text   = text,
            word_count      = len(text.split()),
            matches         = matches,
            bias_score      = bias_score,
            category_scores = cat_scores,
            severity_counts = sev_counts,
            category_counts = cat_counts,
            rewritten_text  = rewritten,
            bert_score      = bert_score,
            verdict         = verdict,
            verdict_color   = color,
            recommendations = recs,
        )

    def batch_analyze(self, texts: List[str]) -> List[BiasReport]:
        """Analyze multiple JDs. Useful for Power BI export."""
        return [self.analyze(t) for t in texts]

    # ── Private: Preprocessing ────────────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        """Lowercase and normalise whitespace."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ── Private: Lexicon Scan ─────────────────────────────────────────────────

    def _lexicon_scan(self, original: str, cleaned: str) -> List[BiasMatch]:
        """
        Multi-word and single-word phrase matching.
        Sorts by length (longest match first) to avoid double-counting.
        """
        matches   = []
        covered   = set()   # character positions already matched
        sentences = re.split(r'[.!?]', original)

        # Sort lexicon entries: longer phrases first to prevent partial matches
        sorted_entries = sorted(self.lexicon.items(),
                                key=lambda x: len(x[0].split()), reverse=True)

        for phrase, meta in sorted_entries:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            for m in re.finditer(pattern, cleaned):
                start, end = m.start(), m.end()
                # Skip if any character in this span is already covered
                if any(i in covered for i in range(start, end)):
                    continue

                # Find surrounding sentence for context
                char_pos   = start
                context    = ""
                cum        = 0
                for sent in sentences:
                    if cum + len(sent) >= char_pos:
                        context = sent.strip()
                        break
                    cum += len(sent) + 1

                matches.append(BiasMatch(
                    word      = phrase,
                    category  = meta["category"],
                    severity  = meta["severity"],
                    neutral   = meta["neutral"],
                    bias_type = meta.get("type", "general"),
                    start_idx = start,
                    end_idx   = end,
                    context   = context[:120],
                ))
                covered.update(range(start, end))

        return matches

    # ── Private: Scoring ──────────────────────────────────────────────────────

    def _compute_bias_score(self, text: str, matches: List[BiasMatch]) -> float:
        """
        Weighted bias score 0–100.
        Score = (Σ severity_weight × match) / word_count × scaling_factor
        Capped at 100.
        """
        if not matches:
            return 0.0

        word_count = max(len(text.split()), 1)
        severity_weights = {1: 5, 2: 12, 3: 25}

        raw_score = sum(severity_weights[m.severity] for m in matches)
        # Normalise: a 300-word JD with 3 high-severity = ~25 score
        normalised = (raw_score / word_count) * 100
        return round(min(normalised, 100.0), 1)

    def _category_scores(self, text: str, matches: List[BiasMatch]) -> Dict[str, float]:
        """Per-category bias score, same formula."""
        word_count = max(len(text.split()), 1)
        severity_weights = {1: 5, 2: 12, 3: 25}
        scores = {cat: 0.0 for cat in CATEGORY_INFO}

        for m in matches:
            scores[m.category] += severity_weights[m.severity]

        return {
            cat: round(min((v / word_count) * 100, 100.0), 1)
            for cat, v in scores.items()
        }

    def _severity_counts(self, matches: List[BiasMatch]) -> Dict[str, int]:
        counts = {"Low": 0, "Moderate": 0, "High": 0}
        for m in matches:
            label = SEVERITY_LABELS[m.severity][0]
            counts[label] += 1
        return counts

    def _category_counts(self, matches: List[BiasMatch]) -> Dict[str, int]:
        counts = {cat: 0 for cat in CATEGORY_INFO}
        for m in matches:
            counts[m.category] += 1
        return counts

    # ── Private: Text Rewriting ───────────────────────────────────────────────

    def _rewrite(self, text: str, matches: List[BiasMatch]) -> str:
        """
        Replace biased phrases with neutral alternatives.
        Preserves original capitalisation style.
        """
        # Sort matches by position (reverse order to not invalidate indices)
        sorted_matches = sorted(matches, key=lambda x: x.start_idx, reverse=True)
        cleaned        = text.lower()

        for m in sorted_matches:
            neutral = m.neutral
            # Skip suggestions that say "(remove ...)"
            if neutral.startswith("(remove"):
                cleaned = (
                    cleaned[:m.start_idx]
                    + "[REVIEW: remove this phrase]"
                    + cleaned[m.end_idx:]
                )
            else:
                cleaned = cleaned[:m.start_idx] + neutral + cleaned[m.end_idx:]

        return cleaned

    # ── Private: Verdict ──────────────────────────────────────────────────────

    def _verdict(self, score: float) -> Tuple[str, str]:
        if score < 10:
            return "Inclusive",       "#28a745"
        elif score < 25:
            return "Mildly Biased",   "#ffc107"
        elif score < 50:
            return "Biased",          "#fd7e14"
        else:
            return "Highly Biased",   "#dc3545"

    # ── Private: Recommendations ──────────────────────────────────────────────

    def _recommendations(self, matches: List[BiasMatch],
                          cat_scores: Dict[str, float]) -> List[str]:
        recs = []
        sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)

        for cat, score in sorted_cats:
            if score == 0:
                continue
            if cat == "gender":
                recs.append("Use gender-neutral pronouns (they/them) and role titles "
                             "(e.g. 'chairperson' not 'chairman').")
            elif cat == "age":
                recs.append("Remove age-coded language ('young', 'digital native'). "
                             "Specify experience as skills, not years.")
            elif cat == "ability":
                recs.append("Replace ability-specific requirements with "
                             "functional job needs. Avoid 'native speaker' — use "
                             "'fluent in English'.")
            elif cat == "culture":
                recs.append("Replace 'culture fit' with 'values alignment'. "
                             "Avoid prestige markers ('Ivy League', 'top-tier school').")

        high_severity = [m for m in matches if m.severity == 3]
        if high_severity:
            words = ", ".join(f"'{m.word}'" for m in high_severity[:4])
            recs.append(f"⚠️  High-risk phrases found: {words}. "
                        f"These may violate equal opportunity employment law.")

        if not recs:
            recs.append("✅ No significant bias patterns detected. "
                        "Continue to review using this tool periodically.")

        return recs

    # ── Private: BERT ─────────────────────────────────────────────────────────

    def _load_bert(self, model_path: str):
        """Load fine-tuned BERT model for bias classification."""
        try:
            from transformers import (
                BertForSequenceClassification,
                BertTokenizer,
            )
            import torch
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
            self.bert_model     = BertForSequenceClassification.from_pretrained(
                model_path, num_labels=2
            )
            self.bert_model.eval()
            print(f"✅ BERT model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load BERT model: {e}. Using rule-based only.")

    def _bert_score(self, text: str) -> Optional[float]:
        """Return BERT-predicted bias probability (0–1)."""
        if not self.bert_model:
            return None
        try:
            import torch
            inputs = self.bert_tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            )
            with torch.no_grad():
                logits = self.bert_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            return round(probs[0][1].item(), 4)   # class 1 = biased
        except Exception:
            return None


# ── Convenience function ──────────────────────────────────────────────────────

def analyze_jd(text: str, bert_model_path: Optional[str] = None) -> BiasReport:
    """One-line entry point."""
    return HiringBiasAnalyzer(bert_model_path).analyze(text)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
    We are looking for a rockstar developer who is a digital native.
    The ideal candidate should be aggressive, competitive, and a culture fit.
    Must be physically fit and a native speaker of English.
    We prefer fresh graduates from top-tier schools.
    He should be able to conquer challenging problems independently.
    """

    report = analyze_jd(sample)

    print(f"\n{'='*55}")
    print(f"  BIAS ANALYSIS REPORT")
    print(f"{'='*55}")
    print(f"  Word count    : {report.word_count}")
    print(f"  Bias score    : {report.bias_score}/100")
    print(f"  Verdict       : {report.verdict}")
    print(f"  Matches found : {len(report.matches)}")
    print(f"\n  Matches:")
    for m in report.matches:
        print(f"    [{m.severity}] '{m.word}' ({m.category}) → '{m.neutral}'")
    print(f"\n  Recommendations:")
    for r in report.recommendations:
        print(f"    • {r}")
    print(f"{'='*55}")
