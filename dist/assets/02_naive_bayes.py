"""Experiment 2: numerically stable Gaussian Naive Bayes from scratch."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(globals().get("__file__", "Computing_Lab1.ipynb")).resolve().parent
# Last column is the target; every preceding column must be a numeric feature.
DATA_FILES = [BASE_DIR / "data" / "iris.csv", BASE_DIR / "data" / "wine.csv"]


class GaussianNBFromScratch:
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or not len(X) or len(X) != len(y) or not np.isfinite(X).all():
            raise ValueError("Expected finite 2-D features and matching labels.")
        self.classes_ = np.unique(y)
        self.means_ = np.array([X[y == label].mean(axis=0) for label in self.classes_])
        variances = np.array([X[y == label].var(axis=0) for label in self.classes_])
        # A floor handles constant features, including an entirely constant dataset.
        epsilon = max(1e-9 * float(X.var(axis=0).max()), 1e-12)
        self.variances_ = variances + epsilon
        self.log_priors_ = np.log([np.mean(y == label) for label in self.classes_])
        return self

    def joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        # Evaluate log Gaussian densities directly; never take log(exp(...)).
        return self.log_priors_ - 0.5 * np.sum(
            np.log(2 * np.pi * self.variances_)
            + (X[:, None, :] - self.means_) ** 2 / self.variances_, axis=2
        )

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]


def run_dataset(path):
    data = pd.read_csv(path, na_values=["?", "", "NA"])
    data = data.dropna(subset=[data.columns[-1]])
    X = data.iloc[:, :-1].apply(pd.to_numeric, errors="raise")
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Learn replacement values from training rows only.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    model = GaussianNBFromScratch().fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\nDataset: {path.name} | train={len(y_train)}, test={len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix (class order:", list(model.classes_), ")")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))
    print(classification_report(y_test, predictions, zero_division=0))
    print(pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": predictions}).head().to_string(index=False))
    return model, X_test, np.asarray(y_test), predictions


def main():
    for path in DATA_FILES:
        run_dataset(path)


if __name__ == "__main__":
    main()
