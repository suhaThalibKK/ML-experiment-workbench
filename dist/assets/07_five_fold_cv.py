"""Experiment 7: five stratified folds, with a fresh decision tree per fold."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(globals().get("__file__", "Computing_Lab1.ipynb")).resolve().parent
DATA_FILE = BASE_DIR / "data" / "iris.csv"


def main():
    data = pd.read_csv(DATA_FILE)
    target = next((c for c in data if c.lower() in {"species", "variety", "class"}), data.columns[-1])
    X = data.drop(columns=[c for c in data if c == target or c.lower() == "id"])
    X = X.apply(pd.to_numeric, errors="raise")
    y = data[target]
    if y.value_counts().min() < 5:
        raise ValueError("Each class needs at least five samples for five stratified folds.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    validation_count = np.zeros(len(data), dtype=int)
    for fold, (train_indices, test_indices) in enumerate(cv.split(X, y), start=1):
        # sklearn's entropy decision tree uses binary CART splits; it is not ID3.
        model = DecisionTreeClassifier(criterion="entropy", random_state=42)
        model.fit(X.iloc[train_indices], y.iloc[train_indices])
        predictions = model.predict(X.iloc[test_indices])
        actual = y.iloc[test_indices]
        validation_count[test_indices] += 1
        rows.append({
            "fold": fold, "accuracy": accuracy_score(actual, predictions),
            "precision_macro": precision_score(actual, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(actual, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(actual, predictions, average="macro", zero_division=0),
        })
    assert np.all(validation_count == 1), "Each row must be validated exactly once."
    scores = pd.DataFrame(rows).set_index("fold")
    print("Per-fold metrics (macro averaging gives each class equal weight):")
    print(scores.to_string(float_format=lambda value: f"{value:.4f}"))
    summary = pd.DataFrame({"mean": scores.mean(), "sample_std": scores.std(ddof=1)})
    print("\nSummary across five folds:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    out = BASE_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "five_fold_scores.csv")
    summary.to_csv(out / "five_fold_summary.csv")
    return scores, validation_count


if __name__ == "__main__":
    main()
