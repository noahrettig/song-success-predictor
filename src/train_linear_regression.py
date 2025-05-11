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

def forward_selection(X_full, y, feature_names, min_improvement=1e-2, max_features=50):
    n, d = X_full.shape
    selected = []
    remaining = list(range(d))
    current_X = np.ones((n, 1))  # Bias term
    sse_history = []

    last_sse = np.inf

    for _ in range(max_features):
        best_sse = np.inf
        best_feature = None
        best_theta = None

        for f in remaining:
            candidate_X = np.hstack([current_X, X_full[:, f].reshape(-1, 1)])
            try:
                theta = np.linalg.solve(candidate_X.T @ candidate_X, candidate_X.T @ y)
            except np.linalg.LinAlgError:
                continue
            sse = sum_square_error(candidate_X, y, theta)
            if sse < best_sse:
                best_sse = sse
                best_feature = f
                best_theta = theta

        improvement = last_sse - best_sse
        if best_feature is None or improvement < min_improvement:
            print(f"*** Stopping: no feature improves SSE by at least {min_improvement} ***")
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        current_X = np.hstack([current_X, X_full[:, best_feature].reshape(-1, 1)])
        sse_history.append(best_sse)
        last_sse = best_sse

        print(f"   Added feature '{feature_names[best_feature]}' | SSE: {best_sse:.2f} | Δ: {improvement:.4f}")

    selected_names = [feature_names[i] for i in selected]
    return selected, selected_names


def main():
    print("Loading data...")
    X_df = pd.read_csv("data/X_processed.csv")
    y_df = pd.read_csv("data/y_labels.csv")["popularity"]
    print("Data loaded!")

    # Drop non-numeric or redundant columns
    excluded_cols = {
        "artist(s)", "song", "album", 
        "genre", "key", "emotion", "text", "release_date", "release_year",
        "popularity", "success_level"
    }
    X_df = X_df.drop(columns=[col for col in X_df.columns if col in excluded_cols])

    # Convert to numpy and standardize
    X_full = X_df.to_numpy(dtype=np.float64)
    y = y_df.to_numpy(dtype=np.float64).reshape(-1, 1)

    X_mean = X_full.mean(axis=0)
    X_std = X_full.std(axis=0) + 1e-8
    X_full = (X_full - X_mean) / X_std
    y = y - y.mean()

    # Forward Selection
    print("Running forward selection...")
    feature_names = X_df.columns.to_list()
    selected_indices, selected_names = forward_selection(X_full, y, feature_names, min_improvement=1.0)
    X_selected = X_full[:, selected_indices]


    # Add "bias" column
    X_selected = np.hstack([np.ones((X_selected.shape[0], 1)), X_selected])

    # Train/test split
    np.random.seed(42)
    indices = np.random.permutation(X_selected.shape[0])
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Ridge Regression with CV
    print("Performing Ridge Regression with CV...")
    lambda_values = np.logspace(-3, 3, 50)
    best_lambda, all_mse = cross_val_ridge(X_train, y_train, lambda_values, k=5)

    theta_best = ridge_theta(X_train, y_train, best_lambda)
    train_mse = mean_squared_error(y_train, X_train @ theta_best)
    test_mse = mean_squared_error(y_test, X_test @ theta_best)

    # Analyze residuals by popularity bins
    y_test_pred = X_test @ theta_best
    residuals = (y_test_pred - y_test).flatten()

    # Create bins
    bins = [-np.inf, 0.3, 0.5, 0.7, np.inf]
    labels = ['Very Low', 'Low', 'Medium', 'High']
    bin_indices = np.digitize(y_test.flatten(), bins)
    bin_labels = [labels[i - 1] for i in bin_indices]

    # Group residuals by bin
    residual_df = pd.DataFrame({
        'residual': residuals,
        'true_popularity': y_test.flatten(),
        'bin': bin_labels
    })

    # Compute metrics
    group_stats = residual_df.groupby('bin').agg(
        count=('residual', 'size'),
        mean_true_popularity=('true_popularity', 'mean'),
        mean_pred_error=('residual', 'mean'),
        std_error=('residual', 'std')
    )

    print("\nResidual analysis by popularity bin:")
    print(group_stats)

    # print(f"\nForward selection chose {len(selected_names)} features:")
    # for i, name in enumerate(selected_names):
    #     print(f"  {i+1:2d}. {name}")

    print(f"Best lambda: {best_lambda:.4f}")
    print(f"Train MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")

    # Plot MSE vs Lambda
    plt.figure(figsize=(8, 5))
    plt.semilogx(lambda_values, all_mse, marker='o')
    plt.axvline(best_lambda, color='red', linestyle='--', label=f'Best λ = {best_lambda:.4f}')
    plt.xlabel("Lambda")
    plt.ylabel("5-Fold Cross-Validated MSE")
    plt.title("MSE vs Lambda (after Forward Selection)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
