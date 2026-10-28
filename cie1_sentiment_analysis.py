"""
CIE 1 Mini-Project: Sentiment Analysis on Movie Reviews using Machine Learning and NLP
======================================================================================
Dataset: IMDb Movie Reviews (2,000 balanced reviews in data/imdb_movie_reviews.csv)
Hardware Target: CPU-optimized for fast execution (< 15 seconds) on Intel Celeron N5100

CIE-1 Preprocessing Requirements:
1. Input Paragraph
2. Convert to lowercase
3. Remove punctuation
4. Tokenize the text
5. Remove stop words
6. Apply stemming (PorterStemmer)
7. Apply lemmatization (WordNetLemmatizer)
8. Display the result after each step
"""

import os
import re
import string
import time
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Configure local NLTK data directory
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(WORKSPACE_DIR, "nltk_data")

import nltk
if os.path.exists(NLTK_DATA_DIR):
    nltk.data.path.insert(0, NLTK_DATA_DIR)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure corpora are available
try:
    STOPWORDS_SET = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS_SET = set(stopwords.words("english"))

STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

# -----------------------------------------------------------------------------
# 1. CIE-1 Preprocessing Step-by-Step Functions
# -----------------------------------------------------------------------------

def to_lowercase(text: str) -> str:
    """Step 1: Convert input text to lowercase."""
    return text.lower()

def remove_punctuation(text: str) -> str:
    """Step 2: Remove punctuation characters while preserving spaces."""
    # Strip HTML tags if present (common in IMDb dataset)
    text = re.sub(r"<.*?>", " ", text)
    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(translator)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", clean_text).strip()

def tokenize_text(text: str) -> List[str]:
    """Step 3: Tokenize the text into individual words."""
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback regex word tokenizer if punkt tab not found
        tokens = re.findall(r"\b\w+\b", text)
    return tokens

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Step 4: Remove English stop words from the token list."""
    return [token for token in tokens if token not in STOPWORDS_SET and len(token) > 1]

def apply_stemming(tokens: List[str]) -> List[str]:
    """Step 5: Apply Porter Stemming to reduce words to their base/root forms."""
    return [STEMMER.stem(token) for token in tokens]

def apply_lemmatization(tokens: List[str]) -> List[str]:
    """Step 6: Apply WordNet Lemmatization with POS fallback to obtain valid dictionary lemmas."""
    return [LEMMATIZER.lemmatize(token) for token in tokens]

def run_preprocessing_steps(paragraph: str) -> Dict[str, Any]:
    """
    Executes all CIE-1 preprocessing steps sequentially and records the intermediate
    output of each step for display and verification.
    """
    step0_input = paragraph.strip()
    step1_lower = to_lowercase(step0_input)
    step2_nopunct = remove_punctuation(step1_lower)
    step3_tokens = tokenize_text(step2_nopunct)
    step4_nostop = remove_stopwords(step3_tokens)
    step5_stemmed = apply_stemming(step4_nostop)
    step6_lemmatized = apply_lemmatization(step4_nostop)

    return {
        "step_0_input": step0_input,
        "step_1_lowercase": step1_lower,
        "step_2_punctuation_removed": step2_nopunct,
        "step_3_tokenized": step3_tokens,
        "step_4_stopwords_removed": step4_nostop,
        "step_5_stemmed": step5_stemmed,
        "step_6_lemmatized": step6_lemmatized,
        "final_cleaned_text": " ".join(step6_lemmatized)
    }

def display_preprocessing_results(steps: Dict[str, Any]) -> None:
    """Formats and prints the result after each CIE-1 step."""
    print("=" * 80)
    print("           CIE-1 PREPROCESSING STEPS - INTERMEDIATE RESULTS")
    print("=" * 80)
    print(f"\n[Step 0] Input Paragraph:\n  {steps['step_0_input']}")
    print(f"\n[Step 1] Lowercase Conversion:\n  {steps['step_1_lowercase']}")
    print(f"\n[Step 2] Punctuation Removal:\n  {steps['step_2_punctuation_removed']}")
    print(f"\n[Step 3] Tokenization ({len(steps['step_3_tokenized'])} tokens):\n  {steps['step_3_tokenized']}")
    print(f"\n[Step 4] Stop Words Removal ({len(steps['step_4_stopwords_removed'])} tokens remaining):\n  {steps['step_4_stopwords_removed']}")
    print(f"\n[Step 5] Stemming (Porter Stemmer):\n  {steps['step_5_stemmed']}")
    print(f"\n[Step 6] Lemmatization (WordNet Lemmatizer):\n  {steps['step_6_lemmatized']}")
    print(f"\n[Final Processed Output]:\n  {steps['final_cleaned_text']}")
    print("=" * 80 + "\n")

# -----------------------------------------------------------------------------
# 2. Dataset Loading & Preprocessing
# -----------------------------------------------------------------------------

def load_dataset(csv_path: str = "data/imdb_movie_reviews.csv") -> pd.DataFrame:
    """
    Loads IMDb movie reviews from the local CSV file.
    Creates and populates the dataset from NLTK if not yet present.
    """
    full_path = os.path.join(WORKSPACE_DIR, csv_path)
    if not os.path.exists(full_path):
        print(f"Dataset not found at {full_path}. Generating from local corpus...")
        from nltk.corpus import movie_reviews
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        data = []
        for fileid in movie_reviews.fileids():
            cat = movie_reviews.categories(fileid)[0]
            sentiment = 1 if cat == "pos" else 0
            text = movie_reviews.raw(fileid)
            data.append({"review": text, "sentiment": sentiment, "label": "Positive" if cat == "pos" else "Negative"})
        df = pd.DataFrame(data)
        df.to_csv(full_path, index=False)
    else:
        df = pd.read_csv(full_path)

    print(f"Loaded {len(df)} reviews from {csv_path}. Class distribution: {df['label'].value_counts().to_dict()}")
    return df

def clean_review_for_ml(text: str) -> str:
    """Fast vectorized cleaning function for training dataset."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS_SET and len(t) > 1]
    return " ".join(tokens)

# -----------------------------------------------------------------------------
# 3. Model Training & Evaluation
# -----------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    """Calculates accuracy, precision, recall, and f1-score."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1-Score": round(float(f1), 4),
        "Confusion Matrix": cm,
    }

def train_and_evaluate(df: pd.DataFrame, models_dir: str = "models") -> Dict[str, Any]:
    """
    Trains TF-IDF Vectorizer, Multinomial Naive Bayes, and Logistic Regression models.
    Saves trained models to disk for fast loading in the Streamlit GUI.
    """
    print("\nPreprocessing dataset for training...")
    t0 = time.time()
    df["cleaned_review"] = df["review"].apply(clean_review_for_ml)
    print(f"Preprocessing completed in {time.time() - t0:.2f} seconds.")

    X = df["cleaned_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples.")

    # 1. Feature Extraction: TF-IDF
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 2. Model 1: Multinomial Naive Bayes
    print("\n--- Training Model 1: Multinomial Naive Bayes ---")
    t_nb = time.time()
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    nb_train_time = time.time() - t_nb
    nb_preds = nb_model.predict(X_test_tfidf)
    nb_metrics = compute_metrics(y_test, nb_preds)
    nb_metrics["Train Time (s)"] = round(nb_train_time, 3)

    # 3. Model 2: Logistic Regression
    print("--- Training Model 2: Logistic Regression ---")
    t_lr = time.time()
    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_train_time = time.time() - t_lr
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_metrics = compute_metrics(y_test, lr_preds)
    lr_metrics["Train Time (s)"] = round(lr_train_time, 3)

    # 4. Display Comparison Table
    comparison_df = pd.DataFrame({
        "Model 1 (Multinomial NB)": [
            nb_metrics["Accuracy"], nb_metrics["Precision"], nb_metrics["Recall"], nb_metrics["F1-Score"], nb_metrics["Train Time (s)"]
        ],
        "Model 2 (Logistic Regression)": [
            lr_metrics["Accuracy"], lr_metrics["Precision"], lr_metrics["Recall"], lr_metrics["F1-Score"], lr_metrics["Train Time (s)"]
        ]
    }, index=["Accuracy", "Precision", "Recall", "F1-Score", "Train Time (s)"])

    print("\n" + "=" * 65)
    print("             MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 65)
    print(comparison_df.to_string())
    print("=" * 65 + "\n")

    # 5. Save Artifacts for Streamlit GUI
    models_path = os.path.join(WORKSPACE_DIR, models_dir)
    os.makedirs(models_path, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(models_path, "tfidf_vectorizer.joblib"))
    joblib.dump(nb_model, os.path.join(models_path, "multinomial_nb.joblib"))
    joblib.dump(lr_model, os.path.join(models_path, "logistic_regression.joblib"))

    summary_metrics = {
        "Multinomial NB": nb_metrics,
        "Logistic Regression": lr_metrics
    }
    with open(os.path.join(models_path, "metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Models and metrics successfully saved to '{models_path}/'")
    return {
        "vectorizer": vectorizer,
        "nb_model": nb_model,
        "lr_model": lr_model,
        "metrics": summary_metrics,
        "comparison_df": comparison_df
    }

# -----------------------------------------------------------------------------
# 4. Single Review Inference Function
# -----------------------------------------------------------------------------

def predict_sentiment(text: str, model_type: str = "lr", artifacts: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Predicts the sentiment of a single review string.
    model_type: 'nb' for Naive Bayes or 'lr' for Logistic Regression.
    """
    if artifacts is None:
        models_path = os.path.join(WORKSPACE_DIR, "models")
        vectorizer = joblib.load(os.path.join(models_path, "tfidf_vectorizer.joblib"))
        if model_type == "nb":
            model = joblib.load(os.path.join(models_path, "multinomial_nb.joblib"))
        else:
            model = joblib.load(os.path.join(models_path, "logistic_regression.joblib"))
    else:
        vectorizer = artifacts["vectorizer"]
        model = artifacts["nb_model"] if model_type == "nb" else artifacts["lr_model"]

    cleaned = clean_review_for_ml(text)
    features = vectorizer.transform([cleaned])
    probs = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))

    label = "Positive" if pred_idx == 1 else "Negative"
    confidence = float(probs[pred_idx])

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "model_used": "Multinomial Naive Bayes" if model_type == "nb" else "Logistic Regression",
        "predicted_sentiment": label,
        "confidence": round(confidence, 4),
        "positive_probability": round(float(probs[1]), 4),
        "negative_probability": round(float(probs[0]), 4),
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("      NLP MINI-PROJECT CIE-1: SENTIMENT ANALYSIS ON MOVIE REVIEWS")
    print("#" * 80)

    # A. Demonstrate all CIE-1 Preprocessing Steps
    sample_paragraph = (
        "The cinematography was breathtaking, and the actors gave an outstanding performance! "
        "However, the pacing in the middle was slightly dragging, but overall it was a wonderful masterpiece."
    )
    print("\n>>> Running CIE-1 Required Preprocessing on Sample Review Paragraph...")
    step_results = run_preprocessing_steps(sample_paragraph)
    display_preprocessing_results(step_results)

    # B. Load Dataset & Train Models
    df = load_dataset()
    artifacts = train_and_evaluate(df)

    # C. Demonstrate Live Sentiment Inference
    print("\n>>> Demonstrating Sentiment Predictions on Sample Test Reviews:")
    test_reviews = [
        "This film was an absolute masterpiece! Sensational directing, brilliant score, and stellar performances.",
        "A complete waste of time. Boring script, lifeless characters, and terrible dialogue throughout.",
        "It was decent with a few funny jokes, though quite predictable and generic overall."
    ]

    for rev in test_reviews:
        res_lr = predict_sentiment(rev, model_type="lr", artifacts=artifacts)
        res_nb = predict_sentiment(rev, model_type="nb", artifacts=artifacts)
        print(f"\nReview: \"{rev}\"")
        print(f"  [Logistic Regression] -> {res_lr['predicted_sentiment']} (Confidence: {res_lr['confidence']*100:.2f}%)")
        print(f"  [Naive Bayes]        -> {res_nb['predicted_sentiment']} (Confidence: {res_nb['confidence']*100:.2f}%)")

    print("\n" + "#" * 80)
    print("                      CIE-1 EXECUTION COMPLETED")
    print("#" * 80 + "\n")
