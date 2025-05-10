# src/train_decision_tree.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def main():
    print("📦 Loading processed data...")
    X = pd.read_csv("data/X_processed.csv")
    y = pd.read_csv("data/y_labels.csv")["success_level"]

    print("✅ Data loaded. Selecting features...")

    # Columns to exclude
    excluded_cols = {'artist(s)', 'song', 'album', 'popularity'}
    
    # Select valid features: all columns not excluded
    feature_cols = [col for col in X.columns if col not in excluded_cols]
    X = X[feature_cols]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 🌳 Train the decision tree
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # 📊 Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Decision Tree Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 🌲 Visualize the tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf, 
        feature_names=X.columns, 
        class_names=sorted(y.unique()), 
        filled=True, 
        rounded=True,
        max_depth=3  # For readability — adjust as needed
    )
    plt.title("Decision Tree (Depth 3 shown)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()