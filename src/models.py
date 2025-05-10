import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

def main():
    print("Loading classification data...")
    X = pd.read_csv("data/X_processed.csv")
    y_df = pd.read_csv("data/y_labels.csv")
    y_class = y_df['success_level']
    y_reg = y_df['popularity']

    # Split for classification
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    # Classification Models
    print("\nTraining classification models...")
    models = {
        "Naive Bayes": train_naive_bayes(X_train, y_train_class),
        "Decision Tree": train_decision_tree(X_train, y_train_class),
        "KNN": train_knn(X_train, y_train_class)
    }

    for name, model in models.items():
        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test_class, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        recall = recall_score(y_test_class, preds, average='macro', zero_division=0)
        print(f"{name} Recall: {recall:.4f}")
        precision = precision_score(y_test_class, preds, average='macro', zero_division=0)
        print(f"{name} Precision: {precision:.4f}")

        # Classification Report
        print(f"{name} Classification Report:")
        print(classification_report(y_test_class, preds, zero_division=0))

        # Confusion Matrix
        # cm = confusion_matrix(y_test_class, preds)
        # plt.figure(figsize=(6, 5))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_test_class.unique()), yticklabels=sorted(y_test_class.unique()))
        # plt.xlabel("Predicted Label")
        # plt.ylabel("True Label")
        # plt.title(f"{name} Confusion Matrix")
        # plt.tight_layout()
        # plt.show()

        # Save model
        path = f"models/{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, path)
        print(f"Saved {name} model to {path}")


    # Split for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    print("\nTraining regression models...")

    # Linear Regression
    lin_model = train_linear_regression(X_train_reg, y_train_reg)
    lin_preds = lin_model.predict(X_test_reg)
    lin_mse = mean_squared_error(y_test_reg, lin_preds)
    print(f"Linear Regression MSE: {lin_mse:.2f}")
    joblib.dump(lin_model, 'models/linear_regression.joblib')
    print("Saved Linear Regression model to models/linear_regression.joblib")


    # Polynomial Regression on numeric features only
    numeric_cols = ['tempo', 'length', 'loudness_(db)']
    X_train_poly = X_train_reg[numeric_cols]
    X_test_poly = X_test_reg[numeric_cols]

    poly_model = train_polynomial_regression(X_train_poly, y_train_reg)
    poly_preds = poly_model.predict(X_test_poly)
    poly_mse = mean_squared_error(y_test_reg, poly_preds)
    print(f"Polynomial Regression MSE: {poly_mse:.2f}")
    joblib.dump(poly_model, 'models/polynomial_regression.joblib')
    print("Saved Polynomial Regression model to models/polynomial_regression.joblib")

    print("\nDone.")

def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(X_train, y_train, sample_weight=weights)
    return model

def train_knn(X_train, y_train, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# To use using only our numerical features.
def train_polynomial_regression(X_train, y_train, degree=2):
    pipeline = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    pipeline.fit(X_train, y_train)
    return pipeline

if __name__ == "__main__":
    main()