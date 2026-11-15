"""Experiment 6: scaled KNN with correct and incorrect predictions."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(globals().get("__file__", "Computing_Lab1.ipynb")).resolve().parent
DATA_FILE = BASE_DIR / "data" / "iris.csv"
K = 3


def main():
    data = pd.read_csv(DATA_FILE)
    target = next((c for c in data if c.lower() in {"species", "variety", "class"}), data.columns[-1])
    X = data.drop(columns=[c for c in data if c == target or c.lower() == "id"])
    X = X.apply(pd.to_numeric, errors="raise")
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    # Scaling is fitted to the training data and reused unchanged on test data.
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    comparison = X_test.copy()
    comparison["actual"] = y_test
    comparison["predicted"] = predictions
    comparison["correct"] = comparison["actual"] == comparison["predicted"]
    print(f"k={K} | train={len(X_train)}, test={len(X_test)}")
    for correct, title in [(True, "Correct predictions"), (False, "Incorrect predictions")]:
        subset = comparison[comparison["correct"] == correct]
        print(f"\n{title}: {len(subset)}")
        print(subset.to_string() if len(subset) else "None")
    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix, class order:", list(model.classes_))
    print(confusion_matrix(y_test, predictions, labels=model.classes_))
    out = BASE_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "knn_predictions.csv", index_label="row")
    return model, comparison


if __name__ == "__main__":
    main()
