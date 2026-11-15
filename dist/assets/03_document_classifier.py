"""Experiment 3: SMS document classification with Multinomial Naive Bayes."""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

BASE_DIR = Path(globals().get("__file__", "Computing_Lab1.ipynb")).resolve().parent
DATA_FILE = BASE_DIR / "data" / "spam.csv"


def main():
    try:
        data = pd.read_csv(DATA_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(DATA_FILE, encoding="latin-1")
    # Also accept the v1/v2 column names used in the reference program.
    data = data.rename(columns={"v1": "label", "v2": "message"})[["label", "message"]]
    data = data.dropna(subset=["label", "message"]).copy()
    data["label"] = data["label"].str.strip().str.lower()
    if not data["label"].isin(["ham", "spam"]).all():
        raise ValueError("Labels must be 'ham' or 'spam'.")
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["message"].ne("")]
    # Remove identical messages before splitting; duplicates can inflate test scores.
    data["key"] = data["message"].str.lower().str.replace(r"\s+", " ", regex=True)
    conflicts = data.groupby("key")["label"].nunique()
    data = data[~data["key"].isin(conflicts[conflicts > 1].index)]
    data = data.drop_duplicates("key")
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )
    # The vocabulary is learned only from training documents.
    model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Unique usable documents: {len(data)} | train={len(X_train)}, test={len(X_test)}")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, pos_label='spam', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, pos_label='spam', zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, predictions, pos_label='spam', zero_division=0):.4f}")
    print("Confusion matrix: rows=actual, columns=predicted; order=[ham, spam]")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))
    messages = ["Congratulations! Claim your free cash prize now", "Can we meet at the library after class?"]
    print("\nNew document predictions:")
    for message, label in zip(messages, model.predict(messages)):
        print(f"{label}: {message}")
    return model, X_train, X_test


if __name__ == "__main__":
    main()
