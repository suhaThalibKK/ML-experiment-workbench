"""Experiment 5: compare K-Means and EM-fitted Gaussian mixture clustering."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(globals().get("__file__", "Computing_Lab1.ipynb")).resolve().parent
DATA_FILE = BASE_DIR / "data" / "iris.csv"
RESULTS_DIR = BASE_DIR / "results"
N_CLUSTERS = 3


def main():
    data = pd.read_csv(DATA_FILE)
    # Species is used only for the external evaluation, never for fitting.
    target = next((c for c in data if c.lower() in {"species", "variety", "class"}), None)
    labels = data[target] if target else None
    features = data.drop(columns=[c for c in data if c == target or c.lower() == "id"])
    X = features.apply(pd.to_numeric, errors="raise").to_numpy()
    if X.shape[1] < 2 or not np.isfinite(X).all():
        raise ValueError("Expected at least two complete numeric feature columns.")
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    # GaussianMixture learns its parameters using Expectation-Maximization.
    gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type="full",
                          n_init=10, max_iter=500, reg_covar=1e-6, random_state=42)
    km_labels = kmeans.fit_predict(X_scaled)
    em_labels = gmm.fit_predict(X_scaled)
    if not gmm.converged_:
        raise RuntimeError("EM did not converge; increase max_iter or inspect the data.")
    scores = {}
    for name, clusters in [("K-Means", km_labels), ("EM (GMM)", em_labels)]:
        count = len(np.unique(clusters))
        if not 1 < count < len(X_scaled):
            raise ValueError("Silhouette requires 2 to n_samples-1 populated clusters.")
        scores[name] = silhouette_score(X_scaled, clusters)
        print(f"{name}: silhouette={scores[name]:.4f}; cluster counts={np.bincount(clusters)}")
        if labels is not None:
            print(f"  Adjusted Rand index vs species: {adjusted_rand_score(labels, clusters):.4f}")
    print(f"EM converged: {gmm.converged_}; iterations: {gmm.n_iter_}")
    print("Larger silhouette on this dataset:", max(scores, key=scores.get))
    print("This comparison does not establish that one method is always better.")
    print("First sample's EM membership probabilities:", gmm.predict_proba(X_scaled[:1])[0].round(4))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, name, clusters, centers in zip(
        axes, ["K-Means", "EM (Gaussian mixture)"], [km_labels, em_labels],
        [kmeans.cluster_centers_, gmm.means_]
    ):
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap="viridis", s=30)
        ax.scatter(centers[:, 0], centers[:, 1], color="#d43f3a", marker="X", s=150, label="Centers")
        ax.set_title(name)
        ax.set_xlabel(features.columns[0] + " (standardized)")
        ax.legend()
    axes[0].set_ylabel(features.columns[1] + " (standardized)")
    fig.suptitle("Iris: first two features shown; clustering uses all four")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "clustering_comparison.png", dpi=160)
    plt.close(fig)
    pd.DataFrame({"kmeans": km_labels, "em_gmm": em_labels}).to_csv(
        RESULTS_DIR / "cluster_assignments.csv", index=False
    )
    return kmeans, gmm, X_scaled


if __name__ == "__main__":
    main()
