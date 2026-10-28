"""
Streamlit GUI Application: CIE-1 NLP Mini-Project
=================================================
Sentiment Analysis on IMDb Movie Reviews
Provides an interactive GUI for:
1. CIE-1 Step-by-Step Preprocessing Visualization
2. Real-Time Sentiment Prediction with Confidence
3. Comparative Model Evaluation Dashboard
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="Movie Sentiment Analysis - CIE 1",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import preprocessing from core module
from cie1_sentiment_analysis import (
    run_preprocessing_steps,
    clean_review_for_ml,
    to_lowercase,
    remove_punctuation,
    tokenize_text,
    remove_stopwords,
    apply_stemming,
    apply_lemmatization
)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")

@st.cache_resource
def load_artifacts():
    """Loads pre-trained vectorizer and models with Streamlit caching."""
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    nb_path = os.path.join(MODELS_DIR, "multinomial_nb.joblib")
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    # If models not found on disk, trigger quick training
    if not (os.path.exists(vec_path) and os.path.exists(nb_path) and os.path.exists(lr_path)):
        from cie1_sentiment_analysis import load_dataset, train_and_evaluate
        df = load_dataset()
        train_and_evaluate(df)

    vectorizer = joblib.load(vec_path)
    nb_model = joblib.load(nb_path)
    lr_model = joblib.load(lr_path)

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return vectorizer, nb_model, lr_model, metrics

vectorizer, nb_model, lr_model, metrics = load_artifacts()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/movie.png", width=70)
    st.title("NLP Mini-Project")
    st.markdown("**CIE - 1 Evaluation**")
    st.markdown("---")
    st.markdown("### Project Highlights")
    st.markdown("- **Task**: Sentiment Analysis")
    st.markdown("- **Dataset**: IMDb Movie Reviews (2,000 samples)")
    st.markdown("- **Models**: Multinomial NB & Logistic Regression")
    st.markdown("- **Preprocessing**: NLTK Full CIE-1 Pipeline")
    st.markdown("---")
    st.caption("Optimized for local CPU execution (Intel Celeron N5100).")

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("🎬 Movie Review Sentiment Analysis & CIE-1 Preprocessing")
st.markdown(
    "Interactive GUI for demonstrating **CIE-1 text preprocessing steps** "
    "and **machine learning sentiment classification** on the IMDb dataset."
)

tab1, tab2, tab3 = st.tabs([
    "🔍 CIE-1 Preprocessing Pipeline",
    "🎯 Real-time Sentiment Predictor",
    "📊 Model Performance & Metrics"
])

# -----------------------------------------------------------------------------
# TAB 1: CIE-1 Preprocessing Pipeline
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Step-by-Step Text Preprocessing")
    st.markdown(
        "Enter any text paragraph below. The system will process it through all **6 mandatory CIE-1 steps** "
        "and display the intermediate output after each stage."
    )

    default_text = (
        "The cinematography was breathtaking, and the actors gave an outstanding performance! "
        "However, the pacing in the middle was slightly dragging, but overall it was a wonderful masterpiece."
    )
    user_paragraph = st.text_area("Input Paragraph / Review:", value=default_text, height=110)

    if st.button("🚀 Run CIE-1 Preprocessing Steps", type="primary", key="btn_prep"):
        steps = run_preprocessing_steps(user_paragraph)

        # Step 0: Input Text
        with st.expander("📌 **Step 0: Original Input Text**", expanded=True):
            st.code(steps["step_0_input"], language="text")

        col1, col2 = st.columns(2)
        with col1:
            # Step 1: Lowercase
            with st.expander("🔤 **Step 1: Lowercase Conversion**", expanded=True):
                st.write("*Converts all characters to lowercase for casing normalization.*")
                st.code(steps["step_1_lowercase"], language="text")

            # Step 3: Tokenization
            with st.expander(f"✂️ **Step 3: Word Tokenization ({len(steps['step_3_tokenized'])} tokens)**", expanded=True):
                st.write("*Splits normalized text into distinct lexical tokens using NLTK.*")
                st.write(steps["step_3_tokenized"])

            # Step 5: Stemming
            with st.expander("🌱 **Step 5: Stemming (Porter Stemmer)**", expanded=True):
                st.write("*Applies Porter Stemming to chop word suffixes down to base stems.*")
                st.write(steps["step_5_stemmed"])

        with col2:
            # Step 2: Remove Punctuation
            with st.expander("🧹 **Step 2: Punctuation & Tag Removal**", expanded=True):
                st.write("*Strips punctuation marks and HTML tags while keeping whitespace clean.*")
                st.code(steps["step_2_punctuation_removed"], language="text")

            # Step 4: Stop Words Removal
            with st.expander(f"🛑 **Step 4: Stop Words Removal ({len(steps['step_4_stopwords_removed'])} tokens kept)**", expanded=True):
                st.write("*Filters out common English stop words (e.g., 'the', 'is', 'was', 'and').*")
                st.write(steps["step_4_stopwords_removed"])

            # Step 6: Lemmatization
            with st.expander("📚 **Step 6: Lemmatization (WordNet Lemmatizer)**", expanded=True):
                st.write("*Reduces tokens to grammatically correct dictionary lemmas.*")
                st.write(steps["step_6_lemmatized"])

        # Stemming vs Lemmatization Comparison Table
        st.markdown("### 🔬 Comparison: Stemming vs. Lemmatization")
        diff_data = []
        for orig, stemmed, lemmatized in zip(steps["step_4_stopwords_removed"], steps["step_5_stemmed"], steps["step_6_lemmatized"]):
            diff_data.append({
                "Filtered Token": orig,
                "Stemmed (Porter)": stemmed,
                "Lemmatized (WordNet)": lemmatized,
                "Different?": "Yes ⚠️" if stemmed != lemmatized else "No ✅"
            })
        st.dataframe(pd.DataFrame(diff_data), use_container_width=True)

        st.success(f"**Final Cleaned Representation for ML:** `{steps['final_cleaned_text']}`")

# -----------------------------------------------------------------------------
# TAB 2: Real-Time Sentiment Predictor
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Live Sentiment Classifier")
    st.markdown("Test sentiment prediction on custom movie reviews using the trained models.")

    # Preset sample reviews for quick demonstration
    col_presets = st.columns(3)
    preset_choice = None
    if col_presets[0].button("👍 Sample: Positive Review"):
        preset_choice = "This film was an absolute masterpiece! Sensational directing, brilliant score, and stellar performances throughout."
    if col_presets[1].button("👎 Sample: Negative Review"):
        preset_choice = "A complete waste of time. Boring script, lifeless characters, and terrible dialogue throughout."
    if col_presets[2].button("⚖️ Sample: Mixed Review"):
        preset_choice = "It had enjoyable moments and decent visuals, but the plot was predictable and dragged in the second half."

    review_input = st.text_area(
        "Enter Movie Review:",
        value=preset_choice if preset_choice else "Sensational acting and an unbelievable screenplay that kept me on the edge of my seat!",
        height=100
    )

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        model_choice = st.selectbox(
            "Select Machine Learning Model:",
            ["Logistic Regression (Accuracy: ~84.3%)", "Multinomial Naive Bayes (Accuracy: ~81.3%)"]
        )
    with col_m2:
        st.write("")
        st.write("")
        predict_clicked = st.button("🎯 Predict Sentiment", type="primary", use_container_width=True)

    if predict_clicked or preset_choice:
        selected_model = lr_model if "Logistic" in model_choice else nb_model
        model_name = "Logistic Regression" if "Logistic" in model_choice else "Multinomial Naive Bayes"

        # Preprocess input and predict
        cleaned = clean_review_for_ml(review_input)
        vec_features = vectorizer.transform([cleaned])
        probs = selected_model.predict_proba(vec_features)[0]
        pred_idx = int(np.argmax(probs))

        sentiment = "Positive" if pred_idx == 1 else "Negative"
        confidence = probs[pred_idx] * 100

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if sentiment == "Positive":
                st.success(f"### Result: POSITIVE 🍿\n**Confidence: {confidence:.2f}%**")
            else:
                st.error(f"### Result: NEGATIVE 🍅\n**Confidence: {confidence:.2f}%**")
            st.caption(f"Model used: **{model_name}**")

        with res_col2:
            st.markdown("#### Probability Breakdown")
            st.write(f"**Positive Probability:** `{probs[1] * 100:.2f}%`")
            st.progress(float(probs[1]))
            st.write(f"**Negative Probability:** `{probs[0] * 100:.2f}%`")
            st.progress(float(probs[0]))

        st.caption(f"**Cleaned Tokens passed to TF-IDF:** `{cleaned}`")

# -----------------------------------------------------------------------------
# TAB 3: Model Performance & Metrics
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Model Evaluation & Performance Comparison")
    st.markdown("Comparison between **Multinomial Naive Bayes** and **Logistic Regression** trained on the IMDb movie dataset.")

    if metrics:
        nb_m = metrics.get("Multinomial NB", {})
        lr_m = metrics.get("Logistic Regression", {})

        # Top summary metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Accuracy", f"{lr_m.get('Accuracy', 0) * 100:.2f}%", "Logistic Regression")
        c2.metric("Best F1-Score", f"{lr_m.get('F1-Score', 0) * 100:.2f}%", "Logistic Regression")
        c3.metric("NB Train Time", f"{nb_m.get('Train Time (s)', 0)}s", "Ultra-fast")
        c4.metric("LR Train Time", f"{lr_m.get('Train Time (s)', 0)}s", "Sub-second")

        st.markdown("---")

        col_tbl, col_cms = st.columns([1, 1])

        with col_tbl:
            st.markdown("### 📋 Metric Comparison Table")
            comp_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Train Time (s)"],
                "Multinomial Naive Bayes": [
                    f"{nb_m.get('Accuracy', 0):.4f}",
                    f"{nb_m.get('Precision', 0):.4f}",
                    f"{nb_m.get('Recall', 0):.4f}",
                    f"{nb_m.get('F1-Score', 0):.4f}",
                    f"{nb_m.get('Train Time (s)', 0):.3f}"
                ],
                "Logistic Regression": [
                    f"{lr_m.get('Accuracy', 0):.4f}",
                    f"{lr_m.get('Precision', 0):.4f}",
                    f"{lr_m.get('Recall', 0):.4f}",
                    f"{lr_m.get('F1-Score', 0):.4f}",
                    f"{lr_m.get('Train Time (s)', 0):.3f}"
                ]
            })
            st.table(comp_df.set_index("Metric"))

        with col_cms:
            st.markdown("### 🎯 Confusion Matrices (Test Set)")
            cm_col1, cm_col2 = st.columns(2)

            with cm_col1:
                st.markdown("**Naive Bayes**")
                cm_nb = nb_m.get("Confusion Matrix", [[0, 0], [0, 0]])
                st.dataframe(pd.DataFrame(cm_nb, columns=["Pred Neg", "Pred Pos"], index=["True Neg", "True Pos"]))

            with cm_col2:
                st.markdown("**Logistic Regression**")
                cm_lr = lr_m.get("Confusion Matrix", [[0, 0], [0, 0]])
                st.dataframe(pd.DataFrame(cm_lr, columns=["Pred Neg", "Pred Pos"], index=["True Neg", "True Pos"]))

    st.markdown("---")
    st.markdown("### ⚙️ Dataset & Pipeline Specifications")
    st.markdown(
        """
        - **Corpus**: IMDb Movie Reviews (2,000 balanced positive and negative reviews).
        - **Vocabulary / Feature Space**: TF-IDF with n-grams (1, 2) capped at 5,000 maximum features.
        - **Hardware Resource Optimization**: Zero heavy neural-network latency; trains in < 1 second and provides instant predictions.
        """
    )
