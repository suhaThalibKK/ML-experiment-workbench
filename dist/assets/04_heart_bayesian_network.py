"""Experiment 4: learn a discrete Bayesian network on Cleveland heart data.

Educational classification experiment; these predictions are not clinical diagnoses.
Uses pgmpy 1.0.0, with training-only discretization and a held-out test set.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(globals().get("__file__", "Computing_Lab1.ipynb")).resolve().parent
DATA_FILE = BASE_DIR / "data" / "heart.csv"
RESULTS_DIR = BASE_DIR / "results"
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Original UCI Cleveland coding, NOT the recoded categories of some Kaggle files.
CATEGORIES = {
    "sex": [0, 1], "cp": [1, 2, 3, 4], "fbs": [0, 1], "restecg": [0, 1, 2],
    "exang": [0, 1], "slope": [1, 2, 3], "ca": [0, 1, 2, 3], "thal": [3, 6, 7],
}
FEATURES = CONTINUOUS + list(CATEGORIES)


def discretize_training_and_test(train, test):
    train_d, test_d = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    states, rules = {}, {}
    for column in FEATURES:
        if train[column].notna().sum() == 0:
            raise ValueError(f"No observed training values in {column}.")
        if column in CONTINUOUS:
            replacement = float(train[column].median())
            train_values = train[column].fillna(replacement)
            test_values = test[column].fillna(replacement)
            cuts = np.unique(np.quantile(train_values, [1 / 3, 2 / 3]))
            train_d[column] = np.searchsorted(cuts, train_values, side="right")
            test_d[column] = np.searchsorted(cuts, test_values, side="right")
            states[column] = list(range(len(cuts) + 1))
            rules[column] = {"median": replacement, "cut_points": cuts.tolist()}
        else:
            replacement = int(train[column].mode().iloc[0])
            train_d[column] = train[column].fillna(replacement).astype(int)
            test_d[column] = test[column].fillna(replacement).astype(int)
            states[column] = CATEGORIES[column]
            rules[column] = {"mode": replacement, "states": CATEGORIES[column]}
    return train_d, test_d, states, rules


def main():
    data = pd.read_csv(DATA_FILE, na_values=["?", "", "NA"])
    required = FEATURES + ["target"]
    if not set(required).issubset(data.columns):
        raise ValueError("Expected UCI Cleveland columns: " + ", ".join(required))
    data = data[required].apply(pd.to_numeric, errors="raise").dropna(subset=["target"])
    if not data["target"].isin([0, 1]).all():
        raise ValueError("target must be 0=absence, 1=presence (map original num > 0 to 1).")
    data["target"] = data["target"].astype(int)
    for column, domain in CATEGORIES.items():
        if not data[column].dropna().isin(domain).all():
            raise ValueError(f"{column} must use original Cleveland states {domain}.")
    if np.isinf(data[FEATURES].to_numpy()).any():
        raise ValueError("Features cannot contain infinite values.")
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["target"])
    train_d, test_d, states, rules = discretize_training_and_test(train, test)
    train_d["target"] = train["target"]
    states["target"] = [0, 1]

    print(f"Records: {len(data)} | train={len(train)}, test={len(test)}")
    print("Continuous features are binned using training-set tertiles.")
    structure = HillClimbSearch(train_d, state_names=states).estimate(
        scoring_method="bic-d", max_indegree=2, max_iter=1000, show_progress=False
    )
    model = DiscreteBayesianNetwork()
    # An edge-only constructor loses isolated variables, potentially the target.
    model.add_nodes_from(train_d.columns)
    model.add_edges_from(structure.edges())
    estimator = BayesianEstimator(model, train_d, state_names=states)
    # pgmpy 1.0.0 also rebuilds the graph inside BayesianEstimator from edges.
    # Restore isolated nodes there so they receive their own prior CPDs.
    estimator.model.add_nodes_from(model.nodes())
    model.add_cpds(*estimator.get_parameters(
        prior_type="BDeu", equivalent_sample_size=5, n_jobs=1
    ))
    if not model.check_model():
        raise RuntimeError("Invalid Bayesian network.")
    print("Learned edges:", sorted(model.edges()))
    print("\nTraining preprocessing rules:")
    for name in CONTINUOUS:
        print(name, rules[name])

    inference = VariableElimination(model)
    predictions, probabilities = [], []
    for _, row in test_d.iterrows():
        evidence = {column: int(row[column]) for column in FEATURES}
        posterior = inference.query(["target"], evidence=evidence, show_progress=False)
        ordered_states = posterior.state_names["target"]
        predictions.append(ordered_states[int(np.argmax(posterior.values))])
        probabilities.append(float(posterior.values[ordered_states.index(1)]))
    print(f"\nHeld-out accuracy: {accuracy_score(test['target'], predictions):.4f}")
    print("Confusion matrix: rows=actual, columns=predicted; order=[0, 1]")
    print(confusion_matrix(test["target"], predictions, labels=[0, 1]))
    print(classification_report(test["target"], predictions, labels=[0, 1], zero_division=0))
    print("First held-out example (raw features):", test.iloc[0][FEATURES].to_dict())
    print("Evidence after preprocessing:", test_d.iloc[0].to_dict())
    print(f"P(target=1 | evidence) = {probabilities[0]:.4f}; prediction = {predictions[0]}")
    print("Educational prediction only; probabilities are not clinically validated.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": test.index, "actual": test["target"].to_numpy(),
                  "predicted": predictions, "p_presence": probabilities}).to_csv(
        RESULTS_DIR / "heart_predictions.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    graph = nx.DiGraph(model)
    levels = [sorted(nodes) for nodes in nx.topological_generations(graph)]
    positions = {node: (i - (len(nodes) - 1) / 2, -depth)
                 for depth, nodes in enumerate(levels) for i, node in enumerate(nodes)}
    nx.draw_networkx(graph, pos=positions, ax=ax,
                     node_color="#c5e5f5", node_size=1700, font_size=9, arrowsize=18)
    ax.set_title("Learned Bayesian network - UCI Cleveland (training data)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "heart_network.png", dpi=160)
    plt.close(fig)
    return model, test_d, states, probabilities


if __name__ == "__main__":
    main()
