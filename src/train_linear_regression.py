# src/train_linear_regression.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    print("📦 Loading data...")
    X = pd.read_csv("data/X_processed.csv")
    y = pd.read_csv("data/y_labels.csv")["popularity"]  # Predicting raw popularity

    # Drop unwanted columns if present
    excluded_cols = {
        "artist(s)", "song", "album", 
        'genre', 'key', 'emotion', 'release_date',  # raw categorical 
        "popularity",'success_level', 'text'        # target or raw text
    }
    X = X.drop(columns=[col for col in X.columns if col in excluded_cols])

    # 🔀 Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ➕ Add bias term (intercept column of 1s)
    ones_train = np.ones((X_train.shape[0], 1))
    X_train_aug = np.concatenate([ones_train, X_train.to_numpy(dtype=np.float64)], axis=1)

    ones_test = np.ones((X_test.shape[0], 1))
    X_test_aug = np.concatenate([ones_test, X_test.to_numpy(dtype=np.float64)], axis=1)

    y_train_np = y_train.to_numpy().reshape(-1, 1)
    y_test_np = y_test.to_numpy().reshape(-1, 1)

    # 🧠 Closed-form solution: theta = (XᵀX)⁻¹Xᵀy
    print("📐 Computing weights using normal equation...")
    # XTX = X_train_aug.T @ X_train_aug
    # XTy = X_train_aug.T @ y_train_np
    theta = np.linalg.pinv(X_train_aug) @ y_train_np    # got singular matrix error, so changed from linalg.solve to this... 

    # 📈 Evaluate on test set
    y_pred = X_test_aug @ theta
    mse = mean_squared_error(y_test_np, y_pred)
    print(f"\n✅ Test Mean Squared Error (MSE): {mse:.2f}")

if __name__ == "__main__":
    main()
