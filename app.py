"""
app.py — Hiring Bias Detector
Interactive Streamlit dashboard for analyzing job descriptions.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import os

from analyzer import HiringBiasAnalyzer
from bias_lexicon import CATEGORY_INFO, SEVERITY_LABELS, BIAS_LEXICON

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Hiring Bias Detector",
    page_icon  = "🔍",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-inclusive   { background:#d4edda; color:#155724; padding:12px 20px; border-radius:8px; font-size:1.3rem; font-weight:700; text-align:center; }
    .verdict-mild        { background:#fff3cd; color:#856404; padding:12px 20px; border-radius:8px; font-size:1.3rem; font-weight:700; text-align:center; }
    .verdict-biased      { background:#fde8d0; color:#7d3b00; padding:12px 20px; border-radius:8px; font-size:1.3rem; font-weight:700; text-align:center; }
    .verdict-high        { background:#f8d7da; color:#721c24; padding:12px 20px; border-radius:8px; font-size:1.3rem; font-weight:700; text-align:center; }
    .flag-high    { background:#f8d7da; border-left:4px solid #dc3545; padding:10px 14px; border-radius:0 6px 6px 0; margin:4px 0; }
    .flag-mod     { background:#fde8d0; border-left:4px solid #fd7e14; padding:10px 14px; border-radius:0 6px 6px 0; margin:4px 0; }
    .flag-low     { background:#fff3cd; border-left:4px solid #ffc107; padding:10px 14px; border-radius:0 6px 6px 0; margin:4px 0; }
    .rec-box      { background:var(--secondary-background-color); border-left:4px solid #4C72B0; padding:10px 14px; border-radius:0 6px 6px 0; margin:4px 0; font-size:0.9rem; }
    .score-ring   { font-size:3rem; font-weight:900; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/diversity.png", width=64)
    st.title("Hiring Bias Detector")
    st.markdown("*Identify and fix biased language in job descriptions.*")
    st.markdown("---")

    st.subheader("⚙️ Settings")
    show_rewrite    = st.toggle("Show rewritten JD", value=True)
    show_context    = st.toggle("Show flagged word context", value=True)
    bert_path       = st.text_input(
        "BERT model path (optional)",
        placeholder="./bert_bias_model",
        help="If you've trained the BERT model, enter the path here for enhanced analysis."
    )

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    This tool detects 4 types of bias in job descriptions:
    - **Gender** — masculine/feminine coded language
    - **Age** — youth-favouring language
    - **Ability** — physical/language requirements
    - **Culture** — exclusionary background markers

    *Built by Mahasweta Talik*
    """)

# ── Initialise analyzer ───────────────────────────────────────────────────────
@st.cache_resource
def get_analyzer(bert_path):
    return HiringBiasAnalyzer(
        bert_model_path=bert_path if bert_path and os.path.exists(bert_path) else None
    )

analyzer = get_analyzer(bert_path if bert_path else "")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Hiring Bias Detector")
st.markdown("*Paste a job description below to detect and fix biased language — powered by NLP + BERT.*")
st.markdown("---")

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_input, tab_batch, tab_lexicon = st.tabs(
    ["📝 Analyze JD", "📊 Batch Analysis", "📖 Bias Lexicon"]
)

# ─── Tab 1: Single JD Analysis ────────────────────────────────────────────────
with tab_input:
    SAMPLE_JD = """We are looking for a rockstar software engineer who is a digital native. The ideal candidate should be aggressive, competitive, and a culture fit for our fast-paced team. He should be physically fit and a native speaker of English. We prefer fresh graduates from top-tier schools who can conquer challenging problems independently. The right person is a driven warrior with no employment gaps and a youthful, dynamic attitude."""

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        jd_text = st.text_area(
            "Paste your job description here",
            height   = 160,
            value    = SAMPLE_JD,
            placeholder="Paste your job description here...",
        )
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        clear_btn   = st.button("🗑️ Clear",                  use_container_width=True)

    if clear_btn:
        jd_text = ""

    if analyze_btn and jd_text.strip():
        with st.spinner("Analyzing for bias patterns..."):
            report = analyzer.analyze(jd_text)

        st.markdown("---")

        # ── Verdict banner ────────────────────────────────────────────────────
        verdict_class_map = {
            "Inclusive"    : "verdict-inclusive",
            "Mildly Biased": "verdict-mild",
            "Biased"       : "verdict-biased",
            "Highly Biased": "verdict-high",
        }
        vclass = verdict_class_map.get(report.verdict, "verdict-mild")
        icon_map = {"Inclusive":"✅","Mildly Biased":"⚠️","Biased":"🚨","Highly Biased":"🔴"}
        st.markdown(
            f'<div class="{vclass}">{icon_map[report.verdict]} {report.verdict} — '
            f'Bias Score: {report.bias_score}/100</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPIs ──────────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Bias Score",    f"{report.bias_score}/100")
        k2.metric("Words",         report.word_count)
        k3.metric("Flags Found",   len(report.matches))
        k4.metric("High Severity", report.severity_counts["High"])
        if report.bert_score is not None:
            k5.metric("BERT Bias Prob", f"{report.bert_score:.0%}")
        else:
            k5.metric("BERT Score", "N/A")

        st.markdown("---")

        # ── Charts row ────────────────────────────────────────────────────────
        col_chart1, col_chart2, col_chart3 = st.columns(3)

        # Bias score gauge
        with col_chart1:
            st.markdown("**Bias Score Breakdown**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            cats   = list(CATEGORY_INFO.keys())
            scores = [report.category_scores[c] for c in cats]
            colors = [CATEGORY_INFO[c]["color"] for c in cats]
            bars   = ax.barh(
                [CATEGORY_INFO[c]["label"] for c in cats],
                scores, color=colors, edgecolor="white"
            )
            for bar, v in zip(bars, scores):
                ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{v:.1f}", va="center", fontsize=9)
            ax.set_xlim(0, max(scores + [20]) * 1.3)
            ax.set_xlabel("Bias Score (0–100)")
            ax.set_title("Category Bias Scores", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Severity pie
        with col_chart2:
            st.markdown("**Flags by Severity**")
            sev_vals   = [report.severity_counts[k] for k in ["High","Moderate","Low"]]
            sev_colors = ["#dc3545","#fd7e14","#ffc107"]
            sev_labels = [f"{k}: {v}" for k, v in zip(["High","Moderate","Low"], sev_vals)]

            if sum(sev_vals) > 0:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                wedges, texts, autotexts = ax.pie(
                    sev_vals, labels=sev_labels, colors=sev_colors,
                    autopct="%1.0f%%", startangle=140,
                    textprops={"fontsize": 9},
                )
                ax.set_title("Severity Distribution", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No bias flags detected 🎉")

        # Category donut
        with col_chart3:
            st.markdown("**Flags by Category**")
            cat_vals   = [report.category_counts[c] for c in cats]
            cat_colors = [CATEGORY_INFO[c]["color"] for c in cats]
            cat_labels = [f"{CATEGORY_INFO[c]['label']}: {v}" for c, v in zip(cats, cat_vals)]

            if sum(cat_vals) > 0:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ax.pie(cat_vals, labels=cat_labels, colors=cat_colors,
                       autopct="%1.0f%%", startangle=90,
                       textprops={"fontsize": 9})
                ax.set_title("Category Distribution", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No bias flags detected 🎉")

        st.markdown("---")

        # ── Flagged words ─────────────────────────────────────────────────────
        st.subheader(f"🚩 Flagged Words ({len(report.matches)} found)")

        if report.matches:
            # Sort by severity descending
            sorted_matches = sorted(report.matches, key=lambda x: x.severity, reverse=True)

            for m in sorted_matches:
                sev_label, _ = SEVERITY_LABELS[m.severity]
                flag_class   = {"High":"flag-high","Moderate":"flag-mod","Low":"flag-low"}[sev_label]
                cat_label    = CATEGORY_INFO[m.category]["label"]

                html = f"""
                <div class="{flag_class}">
                    <strong>'{m.word}'</strong> &nbsp;
                    <span style="background:#e9ecef;padding:2px 8px;border-radius:10px;font-size:0.8rem;">{cat_label}</span>
                    <span style="background:#e9ecef;padding:2px 8px;border-radius:10px;font-size:0.8rem;margin-left:4px;">Severity: {sev_label}</span>
                    <br>
                    <span style="font-size:0.88rem;">💡 Neutral alternative: <em>{m.neutral}</em></span>
                """
                if show_context and m.context:
                    html += f'<br><span style="font-size:0.82rem;color:#6c757d;">Context: "…{m.context}…"</span>'
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.success("✅ No bias flags detected in this job description!")

        st.markdown("---")

        # ── Recommendations ───────────────────────────────────────────────────
        st.subheader("💡 Recommendations")
        for rec in report.recommendations:
            st.markdown(f'<div class="rec-box">{rec}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Rewritten JD ──────────────────────────────────────────────────────
        if show_rewrite:
            st.subheader("✍️ AI-Rewritten (Bias-Reduced) Version")
            st.info("Below is your JD with biased words replaced by neutral alternatives. "
                    "Review and edit before using.")
            st.text_area("Rewritten JD", value=report.rewritten_text, height=180)

            # Download button
            st.download_button(
                label    = "⬇️ Download Rewritten JD",
                data     = report.rewritten_text,
                file_name= "rewritten_jd.txt",
                mime     = "text/plain",
            )

        # ── Power BI export ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📤 Export for Power BI")

        export_data = {
            "Metric"        : ["Bias Score", "Gender Score", "Age Score", "Ability Score", "Culture Score",
                                "Word Count", "Flags Found", "High Severity", "Moderate Severity", "Low Severity"],
            "Value"         : [report.bias_score, report.category_scores["gender"],
                                report.category_scores["age"], report.category_scores["ability"],
                                report.category_scores["culture"], report.word_count,
                                len(report.matches), report.severity_counts["High"],
                                report.severity_counts["Moderate"], report.severity_counts["Low"]],
        }
        export_df = pd.DataFrame(export_data)

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label    = "⬇️ Download Bias Report CSV (for Power BI)",
            data     = csv_buffer.getvalue(),
            file_name= "bias_report.csv",
            mime     = "text/csv",
        )

    elif analyze_btn:
        st.warning("Please paste a job description first.")

# ─── Tab 2: Batch Analysis ────────────────────────────────────────────────────
with tab_batch:
    st.subheader("📊 Batch JD Analysis")
    st.markdown("Upload a CSV with a column named `jd_text` to analyse multiple JDs at once.")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)

        if "jd_text" not in df_upload.columns:
            st.error("CSV must have a column named 'jd_text'")
        else:
            st.info(f"Found {len(df_upload)} job descriptions. Analyzing...")

            with st.spinner("Running batch analysis..."):
                results = []
                progress = st.progress(0)
                for i, row in df_upload.iterrows():
                    r = analyzer.analyze(str(row["jd_text"]))
                    results.append({
                        "jd_text"      : row["jd_text"][:80] + "...",
                        "bias_score"   : r.bias_score,
                        "verdict"      : r.verdict,
                        "flags"        : len(r.matches),
                        "gender_score" : r.category_scores["gender"],
                        "age_score"    : r.category_scores["age"],
                        "ability_score": r.category_scores["ability"],
                        "culture_score": r.category_scores["culture"],
                        "flagged_words": "; ".join([m.word for m in r.matches]),
                    })
                    progress.progress((i + 1) / len(df_upload))

            results_df = pd.DataFrame(results)
            st.success(f"✅ Analysis complete!")

            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Bias Score", f"{results_df['bias_score'].mean():.1f}")
            col2.metric("Highly Biased JDs", len(results_df[results_df["verdict"]=="Highly Biased"]))
            col3.metric("Inclusive JDs", len(results_df[results_df["verdict"]=="Inclusive"]))
            col4.metric("Total Flags", results_df["flags"].sum())

            st.dataframe(results_df, use_container_width=True)

            # Download
            csv_out = results_df.to_csv(index=False)
            st.download_button("⬇️ Download Batch Results (CSV)", csv_out,
                               "batch_bias_results.csv", "text/csv")

    else:
        # Demo with sample data
        st.markdown("**Demo: Sample analysis on 5 JDs**")
        sample_jds = [
            "We need a rockstar ninja coder who is a digital native and culture fit.",
            "Seeking an experienced software engineer with strong communication skills.",
            "Young, energetic salesman needed. Must be aggressive and competitive.",
            "We are an equal-opportunity employer seeking a talented data analyst.",
            "Looking for a driven warrior who can conquer targets. Ivy League preferred.",
        ]

        with st.spinner("Running demo analysis..."):
            demo_results = []
            for jd in sample_jds:
                r = analyzer.analyze(jd)
                demo_results.append({
                    "JD Preview"   : jd[:60] + "...",
                    "Bias Score"   : r.bias_score,
                    "Verdict"      : r.verdict,
                    "Flags"        : len(r.matches),
                    "Flagged Words": ", ".join([m.word for m in r.matches]),
                })

        st.dataframe(pd.DataFrame(demo_results), use_container_width=True)

        # Bar chart of demo scores
        fig, ax = plt.subplots(figsize=(10, 3.5))
        scores = [r["Bias Score"] for r in demo_results]
        colors = ["#dc3545" if s > 50 else "#fd7e14" if s > 25 else
                  "#ffc107" if s > 10 else "#28a745" for s in scores]
        bars = ax.bar(range(len(scores)), scores, color=colors, edgecolor="white")
        ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="High bias threshold")
        ax.axhline(25, color="orange", linestyle="--", alpha=0.5, label="Moderate bias threshold")
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels([f"JD {i+1}" for i in range(len(scores))])
        ax.set_ylabel("Bias Score (0–100)")
        ax.set_title("Bias Scores Across Sample JDs", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─── Tab 3: Bias Lexicon ──────────────────────────────────────────────────────
with tab_lexicon:
    st.subheader("📖 Bias Lexicon Explorer")
    st.markdown(f"**{len(BIAS_LEXICON)} bias patterns** across 4 categories and 3 severity levels.")

    # Filters
    f1, f2 = st.columns(2)
    cat_filter = f1.multiselect(
        "Filter by category",
        options=list(CATEGORY_INFO.keys()),
        default=list(CATEGORY_INFO.keys()),
        format_func=lambda x: CATEGORY_INFO[x]["label"],
    )
    sev_filter = f2.multiselect(
        "Filter by severity",
        options=[1, 2, 3],
        default=[1, 2, 3],
        format_func=lambda x: SEVERITY_LABELS[x][0],
    )

    rows = []
    for word, meta in BIAS_LEXICON.items():
        if meta["category"] in cat_filter and meta["severity"] in sev_filter:
            rows.append({
                "Biased Word/Phrase"   : word,
                "Category"             : CATEGORY_INFO[meta["category"]]["label"],
                "Severity"             : SEVERITY_LABELS[meta["severity"]][0],
                "Type"                 : meta.get("type", "—"),
                "Neutral Alternative"  : meta["neutral"],
            })

    lexicon_df = pd.DataFrame(rows)
    st.markdown(f"*Showing {len(lexicon_df)} of {len(BIAS_LEXICON)} entries*")
    st.dataframe(lexicon_df, use_container_width=True, hide_index=True)

    # Category distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cat_counts = {CATEGORY_INFO[c]["label"]: 0 for c in CATEGORY_INFO}
    sev_counts = {"Low": 0, "Moderate": 0, "High": 0}
    for meta in BIAS_LEXICON.values():
        cat_counts[CATEGORY_INFO[meta["category"]]["label"]] += 1
        sev_counts[SEVERITY_LABELS[meta["severity"]][0]] += 1

    cat_colors = [info["color"] for info in CATEGORY_INFO.values()]
    axes[0].barh(list(cat_counts.keys()), list(cat_counts.values()),
                 color=cat_colors, edgecolor="white")
    axes[0].set_title("Words in Lexicon by Category", fontweight="bold")
    axes[0].set_xlabel("Count")

    sev_colors = ["#ffc107", "#fd7e14", "#dc3545"]
    axes[1].bar(list(sev_counts.keys()), list(sev_counts.values()),
                color=sev_colors, edgecolor="white")
    axes[1].set_title("Words in Lexicon by Severity", fontweight="bold")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
