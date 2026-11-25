"""VeriScope - Fake News / Claim Classification  (Streamlit GUI)

Run:  streamlit run app.py
"""
import html
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from config import (BINARY_FAKE, BINARY_REAL, FIG_DIR, LABEL_DISPLAY, LABEL_ORDER,
                    MODELS_DIR, MODES, REPORTS_DIR)
from src.predict import ClaimClassifier, models_available

st.set_page_config(page_title="VeriScope - Claim Classifier", page_icon="🔍", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      .hero {background: linear-gradient(120deg,#0f2027,#203a43,#2c5364); padding: 1.4rem 1.8rem;
             border-radius: 14px; color: #fff; margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 2.1rem;} .hero p {margin: .3rem 0 0; opacity: .85;}
      .result {border-radius: 14px; padding: 1.2rem 1.5rem; color: #fff; margin: .6rem 0;}
      .result h2 {margin: 0; font-size: 1.9rem;} .result span {opacity: .9;}
      .claimbox {border: 1px solid rgba(128,128,128,.35); border-radius: 10px; padding: .9rem 1rem;
                 line-height: 2; font-size: 1.05rem;}
      .chip {display:inline-block; padding: .15rem .6rem; border-radius: 999px; margin: .15rem .2rem;
             font-size: .85rem; border: 1px solid rgba(128,128,128,.4);}
      .note {font-size: .85rem; opacity: .8;}
    </style>
    """,
    unsafe_allow_html=True,
)

COLOURS = {  # result-card colours per label
    "pants-fire": "#b71c1c", "false": "#e53935", "barely-true": "#ef6c00",
    "half-true": "#f9a825", "mostly-true": "#7cb342", "true": "#2e7d32",
    BINARY_FAKE: "#d32f2f", BINARY_REAL: "#2e7d32",
}

EXAMPLES = [
    "Says the unemployment rate has been cut in half since 2010.",
    "Hospitals in our state are required to turn away patients who cannot pay.",
    "The new tax plan will give every family in America a ten thousand dollar refund next year.",
    "Crime in our city has fallen for six years in a row, according to police records.",
    "Scientists have secretly confirmed that drinking hot water cures all forms of cancer.",
]


@st.cache_resource(show_spinner=False)
def load_model(mode: str) -> ClaimClassifier:
    return ClaimClassifier(mode)


def pretty(label: str) -> str:
    return LABEL_DISPLAY.get(label, label)


def highlight(word_scores, scale=None):
    """Colour original claim words by how strongly they support (green) / oppose (red) the prediction."""
    mx = max([abs(s) for _, s in word_scores] + [1e-9]) if scale is None else scale
    parts = []
    for w, s in word_scores:
        a = min(abs(s) / mx, 1.0) * 0.75
        if abs(s) < 1e-9:
            parts.append(html.escape(w))
        else:
            c = "46,160,67" if s > 0 else "229,57,53"
            parts.append(f"<span style='background:rgba({c},{a:.2f});padding:2px 4px;border-radius:5px'>{html.escape(w)}</span>")
    return " ".join(parts)


def show_image(container, path, caption=None):
    """st.image that works on both old and new Streamlit versions."""
    try:
        container.image(str(path), caption=caption, width="stretch")
    except Exception:
        container.image(str(path), caption=caption, use_container_width=True)


def set_example(text):
    st.session_state["claim"] = text


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    available = [m for m in MODES if models_available(m)]
    if not available:
        st.error("No trained models found. Run `python run_pipeline.py` first.")
        st.stop()
    mode = st.radio("Classification mode", available, format_func=lambda m: MODES[m],
                    help="6 PolitiFact credibility categories, or a simpler Fake-vs-Credible view.")
    clf = load_model(mode)
    mt = clf.metrics
    if mt:
        st.markdown("---")
        st.subheader("Model in use")
        st.write(f"**{mt['best_model']}**")
        st.caption(f"Features: {mt['best_features']}")
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{mt['test']['accuracy']:.1%}")
        c2.metric("Macro F1", f"{mt['test']['f1_macro']:.1%}")
        st.caption(f"Dataset: {mt.get('dataset_source', '-')}")
    st.markdown("---")
    st.caption("Team Group 6 · NLP Capstone · Project 6")

is_demo = bool(mt) and "DEMO" in str(mt.get("dataset_source", "")).upper()

# ----------------------------------------------------------------- header
st.markdown(
    "<div class='hero'><h1>🔍 VeriScope</h1>"
    "<p>Fake News &amp; Claim Classification using NLP · TF-IDF features · classical ML models</p></div>",
    unsafe_allow_html=True,
)
if is_demo:
    st.warning("⚠️ These models were trained on SYNTHETIC demo data - not for the final presentation. "
               "Run `python run_pipeline.py` with the real LIAR dataset.")

tab_predict, tab_perf, tab_eda, tab_data, tab_about = st.tabs(
    ["🧪 Analyze Claim", "📊 Model Performance", "📈 Data Analysis (EDA)", "🗂️ Dataset & Preprocessing", "ℹ️ About"]
)

# ----------------------------------------------------------------- tab 1: predict
with tab_predict:
    st.subheader("Enter a news headline, claim or statement")
    st.markdown("<span class='note'>Try an example:</span>", unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
        col.button(f"Example {i + 1}", key=f"ex{i}", on_click=set_example, args=(ex,))

    claim = st.text_area("Claim / news text", key="claim", height=130,
                         placeholder="e.g. Says the unemployment rate has been cut in half since 2010.")
    go = st.button("🔎 Analyze Claim", type="primary")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if go:
        if len(claim.strip().split()) < 3:
            st.warning("Please enter a claim of at least a few words.")
        else:
            with st.spinner("Analyzing..."):
                res = clf.predict(claim)
            label, conf = res["label"], res["confidence"]
            st.markdown(
                f"<div class='result' style='background:{COLOURS.get(label, '#455a64')}'>"
                f"<span>Predicted category</span><h2>{html.escape(pretty(label))}</h2>"
                f"<span>Model confidence: <b>{conf:.1%}</b></span></div>",
                unsafe_allow_html=True,
            )
            if res["known_terms"] == 0:
                st.info("None of the words in this text were seen during training, so the prediction "
                        "falls back to the most common class and is not reliable.")
            if conf < 0.40 and mode == "multiclass":
                st.caption("Low confidence is normal for the 6-class task - neighbouring categories "
                           "(e.g. 'Half True' vs 'Mostly True') are genuinely hard to separate from text alone.")

            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Probability for each category**")
                order = LABEL_ORDER if mode == "multiclass" else [BINARY_FAKE, BINARY_REAL]
                order = [o for o in order if o in res["probabilities"]]
                pdf = pd.DataFrame({"Category": [pretty(o) for o in order],
                                    "Probability": [res["probabilities"][o] for o in order]}).set_index("Category")
                st.bar_chart(pdf, height=280)
            with right:
                st.markdown("**Why this prediction? (word influence)**")
                st.markdown(f"<div class='claimbox'>{highlight(res['word_scores'])}</div>", unsafe_allow_html=True)
                st.caption("🟩 pushes towards the predicted category · 🟥 pushes against it · plain = ignored / neutral")
                sup = " ".join(f"<span class='chip'>{html.escape(t)}</span>" for t, _ in res["supporting_terms"]) or "-"
                opp = " ".join(f"<span class='chip'>{html.escape(t)}</span>" for t, _ in res["opposing_terms"]) or "-"
                st.markdown(f"**Supporting terms:** {sup}", unsafe_allow_html=True)
                st.markdown(f"**Opposing terms:** {opp}", unsafe_allow_html=True)

            with st.expander("🔧 See the preprocessing the model applied"):
                st.code(f"Original : {claim}\nProcessed: {res['clean_text']}")

            st.info("ℹ️ This is a **model prediction based on writing patterns** learned from labelled PolitiFact claims. "
                    "It does **not** verify facts or establish absolute truth. Always check reliable sources.")
            st.session_state["history"].append({
                "time": datetime.now().strftime("%H:%M:%S"), "mode": MODES[mode], "claim": claim[:120],
                "prediction": pretty(label), "confidence": f"{conf:.1%}"})

    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("**Session history**")
        hdf = pd.DataFrame(st.session_state["history"][::-1])
        st.dataframe(hdf, hide_index=True)
        st.download_button("⬇️ Download history (CSV)", hdf.to_csv(index=False).encode(), "claim_predictions.csv", "text/csv")

# ----------------------------------------------------------------- tab 2: performance
with tab_perf:
    if not mt:
        st.info("Metrics not found.")
    else:
        st.subheader(f"Results on the unseen test set - {MODES[mode]}")
        t = mt["test"]
        c = st.columns(5)
        c[0].metric("Accuracy", f"{t['accuracy']:.1%}")
        c[1].metric("Precision (macro)", f"{t['precision_macro']:.1%}")
        c[2].metric("Recall (macro)", f"{t['recall_macro']:.1%}")
        c[3].metric("F1 (macro)", f"{t['f1_macro']:.1%}")
        c[4].metric("F1 (weighted)", f"{t['f1_weighted']:.1%}")
        st.caption(f"Baselines for context - random guess: {mt['random_baseline_accuracy']:.1%} · "
                   f"always predict the majority class: {mt['majority_baseline_accuracy']:.1%}")

        comp = REPORTS_DIR / f"{mode}_model_comparison.csv"
        if comp.exists():
            st.markdown("**All models × feature sets** (selected by validation macro-F1)")
            cdf = pd.read_csv(comp)
            show = cdf[["features", "model", "val_f1_macro", "test_accuracy", "test_precision_macro",
                        "test_recall_macro", "test_f1_macro"]].rename(columns={
                "features": "Features", "model": "Model", "val_f1_macro": "Val F1",
                "test_accuracy": "Test Acc", "test_precision_macro": "Test Prec",
                "test_recall_macro": "Test Recall", "test_f1_macro": "Test F1"})
            st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[2:]}).highlight_max(
                subset=["Val F1", "Test F1"], color="#2e7d32"), hide_index=True)
        a, b = st.columns(2)
        for col, f in ((a, f"{mode}_model_comparison.png"), (b, f"{mode}_confusion_matrix.png")):
            if (FIG_DIR / f).exists():
                show_image(col, FIG_DIR / f)
        rep = REPORTS_DIR / f"{mode}_classification_report.txt"
        if rep.exists():
            with st.expander("Per-class precision / recall / F1"):
                st.code(rep.read_text())

# ----------------------------------------------------------------- tab 3: EDA
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    summ = REPORTS_DIR / "eda_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        c = st.columns(4)
        c[0].metric("Claims (after cleaning)", f"{s['n_claims']:,}")
        c[1].metric("Vocabulary size", f"{s['vocabulary_size']:,}")
        c[2].metric("Avg words / claim", s["avg_words_raw"])
        c[3].metric("Avg words after preprocessing", s["avg_words_clean"])
    captions = {
        "01_class_distribution.png": "Class distribution - the 6 PolitiFact categories",
        "02_binary_split.png": "Binary view: Fake/Misleading vs Credible",
        "03_length_by_label.png": "Claim length by category",
        "04_length_before_after.png": "Effect of preprocessing on length",
        "05_top_words.png": "Most frequent words per class",
        "06_subjects_speakers.png": "Top subjects & speakers",
        "07_party_vs_label.png": "Credibility mix by party label",
        "08_missing_values.png": "Missing values before cleaning",
    }
    present = [(k, v) for k, v in captions.items() if (FIG_DIR / k).exists()]
    for i in range(0, len(present), 2):
        cols = st.columns(2)
        for col, (fn, cap) in zip(cols, present[i:i + 2]):
            show_image(col, FIG_DIR / fn, cap)

# ----------------------------------------------------------------- tab 4: dataset
with tab_data:
    st.subheader("Dataset: LIAR (PolitiFact fact-checked statements)")
    st.write("Short political statements labelled by professional fact-checkers into six credibility levels. "
             "Source: Wang (2017), *“Liar, Liar Pants on Fire”*, ACL.")
    log = REPORTS_DIR / "cleaning_log.json"
    if log.exists():
        lg = json.loads(log.read_text())
        c = st.columns(4)
        c[0].metric("Raw rows", f"{lg['rows_raw']:,}")
        c[1].metric("Duplicates removed", f"{lg['duplicate_claims_removed']:,}")
        c[2].metric("Empty after preprocessing", lg["empty_after_preprocessing_removed"])
        c[3].metric("Final rows", f"{lg['rows_final']:,}")
        with st.expander("Missing values before cleaning"):
            st.json(lg["missing_before"])
    sample = (REPORTS_DIR.parent / "data" / "processed" / "liar_clean.csv")
    if sample.exists():
        st.markdown("**Sample: original claim → preprocessed text**")
        df = pd.read_csv(sample, usecols=["statement", "clean_text", "label", "split"]).sample(15, random_state=1)
        df["label"] = df["label"].map(pretty)
        st.dataframe(df, hide_index=True)
    st.markdown("""
**Preprocessing pipeline:** lower-casing → remove URLs/HTML/symbols → numbers → `num` token →
contraction handling (`n't` → `not`) → tokenisation → stop-word removal (negations kept) → Porter stemming.

**Feature engineering:** Bag-of-Words, TF-IDF (unigrams) and TF-IDF (unigrams + bigrams) were compared.
""")

# ----------------------------------------------------------------- tab 5: about
with tab_about:
    st.subheader("About this project")
    st.markdown("""
**VeriScope** classifies a news headline or claim into a credibility category based on patterns
in a labelled fact-checking dataset.

**Pipeline:** Dataset collection → Cleaning → Preprocessing → EDA → Feature engineering (BoW / TF-IDF)
→ Model training (Naive Bayes, Logistic Regression, Linear SVM, Random Forest) → Evaluation
(Accuracy, Precision, Recall, F1) → Streamlit GUI.

**Real-world applications:** triage tool for fact-checkers, newsroom assistants, social-media moderation
support, media-literacy education.

**Limitations (be honest!)**
- The model learns *writing style and topic cues*, not facts - it cannot check whether a claim is actually true.
- Trained on US political statements (2007-2016); other domains/time periods may behave differently.
- Distinguishing neighbouring categories (e.g. Half True vs Mostly True) from text alone is very hard
  - published text-only results on LIAR are also modest.
- Predictions are probabilistic and should support, never replace, human fact-checking.

**Future scope:** transformer models (BERT/RoBERTa), adding speaker/context metadata, evidence retrieval
from trusted sources, multilingual support.
""")
