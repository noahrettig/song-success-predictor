# src/train_nonlinear_regression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    print("📦 Loading processed data...")
    X = pd.read_csv("data/X_processed.csv")
    y = pd.read_csv("data/y_labels.csv")["popularity"]

    # 🎯 Keep only useful numeric features
    numeric_cols = ['length', 'tempo', 'loudness_(db)', 'release_month']
    X = X[numeric_cols]

    # 🌀 Add sine/cosine components for release_month (seasonality)
    months = X['release_month'].to_numpy()
    sin_month = np.sin(2 * np.pi * months / 12)
    cos_month = np.cos(2 * np.pi * months / 12)

    # Final feature matrix with bias term and nonlinear seasonal terms
    X_final = np.column_stack([
        np.ones(len(X)),                  # bias
        X['length'].to_numpy(),
        X['tempo'].to_numpy(),
        X['loudness_(db)'].to_numpy(),
        sin_month,
        cos_month
    ])

    # 🔀 Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    # 🧠 Train with normal equation
    print("📐 Fitting nonlinear regression model...")
    theta = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)

    # 📊 Evaluate
    y_pred = X_test @ theta
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n✅ Test Mean Squared Error (MSE): {mse:.2f}")

if __name__ == "__main__":
    main()
