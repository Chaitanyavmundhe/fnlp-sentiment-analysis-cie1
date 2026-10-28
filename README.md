# NLP Mini-Project: Sentiment Analysis on Movie Reviews (CIE-1)

An academic Natural Language Processing mini-project for **CIE-1** performing end-to-end sentiment classification on the **IMDb Movie Reviews dataset**.

## Project Highlights
- **CIE-1 Preprocessing**: Full implementation of all 6 mandatory preprocessing stages with step-by-step intermediate output display.
- **Fast Local Execution**: Optimized for Intel Celeron N5100 CPU (< 15-30 seconds runtime; zero GPU or heavy neural-network requirements).
- **100% Offline & Local**: Self-contained NLTK data and local CSV dataset (`data/imdb_movie_reviews.csv`).
- **Interactive Streamlit GUI**: Web interface to test custom reviews, view intermediate preprocessing steps, and inspect model performance.
- **Academic Jupyter Notebook**: [`cie1_sentiment_analysis.ipynb`](./cie1_sentiment_analysis.ipynb) ready for academic evaluation.

---

## Directory Structure
```text
├── app.py                         # Interactive Streamlit Web GUI
├── cie1_sentiment_analysis.py     # Main Python pipeline (CIE-1 steps + model training)
├── cie1_sentiment_analysis.ipynb  # Academic Jupyter Notebook
├── requirements.txt               # Lightweight Python dependencies
├── data/
│   └── imdb_movie_reviews.csv     # 2,000 balanced IMDb reviews (1,000 pos / 1,000 neg)
├── models/                        # Serialized TF-IDF vectorizer & models (joblib)
│   ├── tfidf_vectorizer.joblib
│   ├── multinomial_nb.joblib
│   ├── logistic_regression.joblib
│   └── metrics.json
└── nltk_data/                     # Local self-contained NLTK corpora
```

---

## CIE-1 Required Preprocessing Steps
1. **Input Paragraph**: Accepts raw text or user input.
2. **Lowercase Conversion**: Case normalization.
3. **Punctuation Removal**: Strips punctuation and HTML markup.
4. **Tokenization**: Words split into discrete lexical tokens.
5. **Stop Words Removal**: Eliminates common English function words.
6. **Stemming**: Porter Stemmer reduces inflected words to stems.
7. **Lemmatization**: WordNet Lemmatizer extracts valid dictionary roots.
8. **Display Intermediate Results**: Each step's output is displayed sequentially.

---

## How to Run

### 1. Run the Python Pipeline Script
Executes all CIE-1 preprocessing steps on sample text, trains models, prints comparison tables, and performs sample predictions:
```bash
./venv/bin/python cie1_sentiment_analysis.py
```

### 2. Launch the Interactive Streamlit GUI
Starts the web application on `http://localhost:8501`:
```bash
./venv/bin/streamlit run app.py
```
- **Tab 1**: CIE-1 Preprocessing Pipeline (enter any text and view each step).
- **Tab 2**: Real-time Sentiment Predictor (live sentiment classification & confidence).
- **Tab 3**: Model Performance & Metrics Dashboard.

### 3. Open the Jupyter Notebook
```bash
./venv/bin/jupyter notebook cie1_sentiment_analysis.ipynb
```

---

## Model Performance Summary

| Metric | Multinomial Naive Bayes | Logistic Regression |
| :--- | :---: | :---: |
| **Accuracy** | 81.25% | **84.25%** |
| **Precision** | 82.05% | **83.09%** |
| **Recall** | 80.00% | **86.00%** |
| **F1-Score** | 81.01% | **84.52%** |
| **Train Time** | ~0.025s | ~0.302s |
