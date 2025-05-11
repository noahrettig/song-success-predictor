# src/train_knn.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def main():
    print("📦 Loading processed data...")
    X = pd.read_csv("data/X_processed.csv")
    y = pd.read_csv("data/y_labels.csv")["success_level"]

    # 🔍 Drop irrelevant metadata columns
    excluded_cols = {'artist(s)', 'song', 'album', 'popularity'}
    X = X.drop(columns=[col for col in X.columns if col in excluded_cols])

    print("🔍 Performing 5-fold cross-validation to find best k (by macro F1)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    avg_f1s = []

    for k in range(1, 31):
        f1_total = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            f1_total += f1_score(y_test, y_pred, average="macro", zero_division=0)
        avg_f1 = f1_total / 5
        avg_f1s.append(avg_f1)

    best_k = np.argmax(avg_f1s) + 1
    print(f"\n✅ Best k: {best_k}")
    print(f"📈 Best average macro F1 score: {avg_f1s[best_k - 1]:.4f}")

    print("\n🎯 Training final KNN model on full training set...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    knn_final.fit(X_train, y_train)
    y_pred_final = knn_final.predict(X_test)

    acc = accuracy_score(y_test, y_pred_final)
    recall = recall_score(y_test, y_pred_final, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred_final, average="macro", zero_division=0)
    print(f"\n✅ Final Test Accuracy: {acc:.4f}")
    print(f"🔁 Final Test Macro Recall: {recall:.4f}")
    print(f"🎯 Final Test Macro F1 Score: {f1:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_final, zero_division=0))

if __name__ == "__main__":
    main()
