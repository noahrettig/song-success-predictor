# src/train_naive_bayes.py
# Kinda garbage. Abandoning this model for the time being. 

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

def main():
    # 🔢 Load binned manual feature dataset
    df = pd.read_csv("data/manual_features.csv")

    # 🎯 Target and features
    target_col = "success_level"
    feature_cols = [col for col in df.columns if col not in [target_col, "popularity"]]

    class_labels = df[target_col].unique()

    # 📊 Compute P(class)
    priors = df[target_col].value_counts(normalize=True).to_dict()

    # 🛠️ Build conditional probability tables P(feature_value | class)
    cond_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # feature -> class -> value -> P

    for feature in feature_cols:
        ctab = pd.crosstab(df[target_col], df[feature])
        for cls in ctab.index:
            row_sum = ctab.loc[cls].sum()
            for val in ctab.columns:
                # Laplace smoothing
                cond_probs[feature][cls][val] = (ctab.loc[cls, val] + 1) / (row_sum + len(ctab.columns))

    # 🧠 Predict for each row by multiplying P(class) * ∏ P(feature=value | class)
    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        posteriors = {}
        for cls in class_labels:
            p = priors.get(cls, 1e-6)
            for feature in feature_cols:
                val = row[feature]
                p *= cond_probs[feature][cls].get(val, 1e-6)  # fallback smoothing
            posteriors[cls] = p
        pred = max(posteriors, key=posteriors.get)
        if pred == row[target_col]:
            correct += 1

    acc = correct / total
    print(f"\n✅ Accuracy using all features: {acc:.4f} ({correct}/{total} correct)")

if __name__ == "__main__":
    main()