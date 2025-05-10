import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def plot_tree_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support .feature_importances_")
    
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(8, 6))
    importances.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title("Tree-based Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

def plot_linear_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not support .coef_")
    
    coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    coef_series = pd.Series(np.abs(coef), index=feature_names)
    coef_series = coef_series.sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(8, 6))
    coef_series.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title("Linear Model Feature Influence (|coefficients|)")
    plt.xlabel("Absolute Value of Coefficient")
    plt.tight_layout()
    plt.show()

def plot_permutation_importance(model, X_test, y_test, top_n=15):
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=X_test.columns)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    importances.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title("Permutation Feature Importance")
    plt.xlabel("Mean Importance Over Shuffles")
    plt.tight_layout()
    plt.show()

def main():
    print("📦 Loading data and models...")
    X = pd.read_csv("data/X_processed.csv")
    y = pd.read_csv("data/y_labels.csv")["success_level"]

    # Load models
    models = {
        "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
        "Decision Tree": joblib.load("models/decision_tree.joblib"),
        "KNN": joblib.load("models/knn.joblib"),
        "Linear Regression": joblib.load("models/linear_regression.joblib"),
        "Polynomial Regression": joblib.load("models/polynomial_regression.joblib"),
    }

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n📊 Analyzing Feature Importances:")

    for name, model in models.items():
        if name != 'Naive Bayes': continue #TEMP: Only analyzes NB, since script dies on my PC if I run it with everything.

        print(f"\n➡️ {name}")
        try:
            if isinstance(model, DecisionTreeClassifier):
                plot_tree_feature_importance(model, X_train.columns)

            elif isinstance(model, LinearRegression):
                plot_linear_feature_importance(model, X_train.columns)

            elif isinstance(model, Pipeline):
                # Assume Polynomial Regression is a pipeline
                print("(Polynomial Regression — using linear model inside pipeline)")
                linear = model.named_steps['linearregression']
                poly_features = model.named_steps['polynomialfeatures'].get_feature_names_out(X_train.columns[:3])  # assumes 3 features
                plot_linear_feature_importance(linear, poly_features)

            else:
                print("(Using permutation importance)")
                plot_permutation_importance(model, X_test, y_test)

        except Exception as e:
            print(f"⚠️ Could not analyze {name}: {e}")

if __name__ == "__main__":
    main()

