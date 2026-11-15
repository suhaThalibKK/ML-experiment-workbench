"""Experiment 1: categorical ID3, implemented from scratch."""
from collections import Counter
from dataclasses import dataclass, field
from math import log2
from pathlib import Path

import pandas as pd

BASE_DIR = Path(globals().get("__file__", "Computing_Lab1.ipynb")).resolve().parent
DATA_FILE = BASE_DIR / "data" / "weather.csv"


@dataclass
class Node:
    prediction: str  # Majority class at this node; also the unseen-value fallback.
    attribute: str | None = None
    children: dict = field(default_factory=dict)


def entropy(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    return -sum((n / total) * log2(n / total) for n in counts.values()) if total else 0.0


def information_gain(data, attribute, target):
    weighted_entropy = sum(
        len(group) / len(data) * entropy(group[target])
        for _, group in data.groupby(attribute, sort=True)
    )
    return entropy(data[target]) - weighted_entropy


def id3(data, attributes, target="play"):
    if data.empty:
        raise ValueError("ID3 needs nonempty training data.")
    # Alphabetical tie-breaking makes the result reproducible.
    counts = Counter(data[target])
    majority = sorted(counts, key=lambda label: (-counts[label], str(label)))[0]
    node = Node(prediction=majority)
    if len(counts) == 1 or not attributes:
        return node
    best = max(attributes, key=lambda name: information_gain(data, name, target))
    # Split even when gain is zero: later attributes can resolve an XOR pattern.
    node.attribute = best
    remaining = [name for name in attributes if name != best]
    for value, group in data.groupby(best, sort=True):
        node.children[value] = id3(group, remaining, target)
    return node


def classify(node, sample):
    while node.attribute is not None:
        child = node.children.get(sample.get(node.attribute))
        if child is None:
            return node.prediction
        node = child
    return node.prediction


def print_tree(node, indent=""):
    if node.attribute is None:
        print(indent + "Predict: " + str(node.prediction))
        return
    print(indent + node.attribute + " [fallback=" + str(node.prediction) + "]")
    for value, child in node.children.items():
        print(indent + "  " + str(value) + ":")
        print_tree(child, indent + "    ")


def main():
    # Read the existing file; do not overwrite it or read before creating it.
    data = pd.read_csv(DATA_FILE, dtype=str)
    target = "play"
    features = [name for name in data if name not in {target, "id"}]
    if target not in data or not features or data[features + [target]].isna().any().any():
        raise ValueError("Expected complete categorical features and a 'play' target.")
    print("Information gain:")
    for feature in features:
        print(f"  {feature}: {information_gain(data, feature, target):.4f}")
    root = id3(data, features, target)
    print("\nDecision tree:")
    print_tree(root)
    sample = {"outlook": "sunny", "temperature": "hot", "humidity": "normal", "windy": "False"}
    print("\nNew sample:", sample)
    print("Predicted label:", classify(root, sample))
    correct = sum(classify(root, row) == row[target] for row in data.to_dict("records"))
    print(f"Training fit (not a test accuracy): {correct}/{len(data)}")
    return root


if __name__ == "__main__":
    main()
