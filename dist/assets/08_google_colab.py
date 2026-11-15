"""Experiment 8: Python, NumPy, pandas, plots, files and ML in Google Colab.

This script also runs locally. Colab-only upload/Drive/download examples are in
the notebook's final section and are optional interactive steps.
"""
import platform
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

BASE_DIR = Path(globals().get("__file__", "Computing_Lab1.ipynb")).resolve().parent


def main():
    print("Python:", platform.python_version())
    print("Operating system:", platform.system())
    print("Hello, Google Colab!")
    matrix = np.array([[1, 2], [3, 4]])
    print("\nNumPy matrix:\n", matrix)
    print("Matrix product A @ A:\n", matrix @ matrix)
    frame = pd.DataFrame({"student": ["A", "B", "C"], "marks": [78, 85, 92]})
    print("\npandas DataFrame:\n", frame.to_string(index=False))
    print("Mean mark:", frame["marks"].mean())
    out = BASE_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "colab_demo.csv", index=False)
    reloaded = pd.read_csv(out / "colab_demo.csv")
    print("CSV saved and read back:", reloaded.equals(frame))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(frame["student"], frame["marks"], color="#277da1")
    ax.set(xlabel="Student", ylabel="Marks", title="Google Colab: plotting example", ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(out / "colab_demo.png", dpi=160)
    plt.close(fig)

    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42, stratify=dataset.target
    )
    start = perf_counter()
    model = GaussianNB().fit(X_train, y_train)
    print(f"\nIris test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(f"Fit and prediction time: {perf_counter() - start:.4f} seconds (machine-dependent)")
    print("Results saved in:", out)
    print("These experiments run on a CPU; a GPU is not required.")


if __name__ == "__main__":
    main()
