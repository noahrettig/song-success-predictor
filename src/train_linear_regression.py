# src/train_linear_regression.py

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def sum_square_error(x, y, theta):
    y_hat = x @ theta
    return np.sum((y - y_hat) ** 2)

def ridge_theta(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)

def cross_val_ridge(X, y, lambdas, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    avg_mse = []

    for lam in lambdas:
        mse_folds = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            theta = ridge_theta(X_train, y_train, lam)
            y_pred = X_val @ theta
            mse_folds.append(mean_squared_error(y_val, y_pred))
        avg_mse.append(np.mean(mse_folds))

    return lambdas[np.argmin(avg_mse)], avg_mse

def main():
    print("📦 Loading data...")
    X_df = pd.read_csv("data/X_processed.csv")
    y_df = pd.read_csv("data/y_labels.csv")["popularity"]

    # Drop non-numeric or redundant columns
    excluded_cols = {
        "artist(s)", "song", "album", 
        "genre", "key", "emotion", "text", "release_date", 'release_year' 
        "popularity", "success_level"
    }
    X_df = X_df.drop(columns=[col for col in X_df.columns if col in excluded_cols])

    # Convert to numpy and standardize
    X = X_df.to_numpy(dtype=np.float64)
    y = y_df.to_numpy(dtype=np.float64).reshape(-1, 1)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8  # prevent divide-by-zero
    X = (X - X_mean) / X_std
    y_mean = y.mean()
    y = y - y_mean

    # Add bias column
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Split into train/test
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Cross-validate ridge regression 
    print("🔁 Performing manual Ridge Regression with CV...")
    lambda_values = np.logspace(-3, 3, 50)
    best_lambda, all_mse = cross_val_ridge(X_train, y_train, lambda_values, k=5)

    theta_best = ridge_theta(X_train, y_train, best_lambda)
    train_mse = mean_squared_error(y_train, X_train @ theta_best)
    test_mse = mean_squared_error(y_test, X_test @ theta_best)

    print(f"\n✅ Best lambda: {best_lambda:.4f}")
    print(f"📉 Train MSE: {train_mse:.2f}")
    print(f"📊 Test MSE: {test_mse:.2f}")

    # 📈 Plotting
    plt.figure(figsize=(8, 5))
    plt.semilogx(lambda_values, all_mse, marker='o')
    plt.axvline(best_lambda, color='red', linestyle='--', label=f'Best λ = {best_lambda:.4f}')
    plt.xlabel("Lambda")
    plt.ylabel("5-Fold Cross-Validated MSE")
    plt.title("MSE vs Lambda")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
