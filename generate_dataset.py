"""
generate_dataset.py
Generates a synthetic dataset of job descriptions with varying bias levels.
Used for EDA, Power BI export, and BERT fine-tuning data augmentation.
"""

import random
import pandas as pd
import numpy as np
from analyzer import HiringBiasAnalyzer

random.seed(42)
np.random.seed(42)

# ── JD Templates ──────────────────────────────────────────────────────────────

INCLUSIVE_TEMPLATES = [
    "We are looking for a skilled {role} with experience in {skills}. The ideal candidate has strong communication skills and a collaborative mindset.",
    "Join our team as a {role}. You will be responsible for {task} and working closely with cross-functional teams.",
    "We are an equal-opportunity employer seeking a {role} with expertise in {skills}. All backgrounds welcome.",
    "Our team is growing and we need a talented {role}. We value diverse perspectives and encourage everyone to apply.",
    "Experienced {role} needed. You will lead {task} and mentor junior team members. Flexible work arrangements available.",
    "We are hiring a {role} to help us {task}. Strong problem-solving skills and the ability to communicate complex ideas clearly are essential.",
    "The {role} will collaborate with stakeholders to {task}. We welcome candidates from non-traditional backgrounds.",
]

BIASED_TEMPLATES = [
    "We need a rockstar {role} who is a digital native. Must be aggressive and competitive. Culture fit is essential.",
    "Looking for a young, energetic {role} to conquer {task}. He should be a ninja with {skills}.",
    "Dynamic {role} needed — fresh graduate preferred. Must be physically fit and a native English speaker.",
    "We want a driven warrior to dominate {task}. Strong, fearless champions only. From an Ivy League or top-tier school.",
    "Our fast-paced startup needs a tech-savvy {role} with 0-2 years of experience. Brotherhood culture, great vibe.",
    "Hiring a {role} guru who is youthful and hip. Must boast expertise in {skills}. No employment gaps.",
    "Seeking an aggressive {role} with a dominant personality. Salesman mindset, manpower addition to our rockstar team.",
]

ROLES  = ["Software Engineer", "Data Analyst", "Product Manager", "Marketing Manager",
          "HR Manager", "Business Analyst", "Data Scientist", "UX Designer",
          "DevOps Engineer", "Sales Manager"]

SKILLS = ["Python and SQL", "machine learning and deep learning", "data visualisation and storytelling",
          "cloud architecture", "agile project management", "statistical analysis",
          "NLP and text analytics", "React and Node.js", "stakeholder management", "Excel and Power BI"]

TASKS  = ["drive product roadmap", "analyse customer data", "build scalable pipelines",
          "lead cross-functional projects", "optimise marketing campaigns",
          "develop ML models", "improve user experience", "manage engineering teams"]

INDUSTRIES = ["fintech", "e-commerce", "healthcare", "edtech", "SaaS", "consulting",
               "media", "logistics", "banking", "retail"]


def generate_jd(biased: bool) -> str:
    template  = random.choice(BIASED_TEMPLATES if biased else INCLUSIVE_TEMPLATES)
    role      = random.choice(ROLES)
    skills    = random.choice(SKILLS)
    task      = random.choice(TASKS)
    return template.format(role=role, skills=skills, task=task)


def generate_dataset(n: int = 200) -> pd.DataFrame:
    """
    Generate n JDs (50% biased, 50% inclusive).
    Returns DataFrame with raw text, label, and bias analysis columns.
    """
    analyzer = HiringBiasAnalyzer()
    records  = []

    print(f"Generating {n} job descriptions...")

    for i in range(n):
        biased = (i % 2 == 0)
        text   = generate_jd(biased)
        report = analyzer.analyze(text)

        records.append({
            "jd_id"            : i + 1,
            "industry"         : random.choice(INDUSTRIES),
            "role"             : random.choice(ROLES),
            "jd_text"          : text,
            "true_label"       : 1 if biased else 0,
            "bias_score"       : report.bias_score,
            "verdict"          : report.verdict,
            "match_count"      : len(report.matches),
            "gender_score"     : report.category_scores["gender"],
            "age_score"        : report.category_scores["age"],
            "ability_score"    : report.category_scores["ability"],
            "culture_score"    : report.category_scores["culture"],
            "high_severity"    : report.severity_counts["High"],
            "moderate_severity": report.severity_counts["Moderate"],
            "low_severity"     : report.severity_counts["Low"],
            "word_count"       : report.word_count,
            "flagged_words"    : "; ".join([m.word for m in report.matches]),
            "rewritten_text"   : report.rewritten_text,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} done...")

    df = pd.DataFrame(records)
    print(f"✅ Dataset generated: {len(df)} rows")
    return df


if __name__ == "__main__":
    df = generate_dataset(200)

    # Save for Power BI and analysis
    df.to_csv("jd_bias_dataset.csv", index=False)
    print(f"\nDataset saved to jd_bias_dataset.csv")
    print(f"\nClass distribution:")
    print(df["verdict"].value_counts())
    print(f"\nAverage bias score: {df['bias_score'].mean():.2f}")
    print(f"Most common flagged category: {df[['gender_score','age_score','ability_score','culture_score']].mean().idxmax()}")
