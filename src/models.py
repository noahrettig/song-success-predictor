import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def main():
    print("📦 Loading classification data...")
    X = pd.read_csv("data/X_processed.csv")
    y_df = pd.read_csv("data/y_labels.csv")
    y_class = y_df['success_level']
    y_reg = y_df['popularity']

    # 🔀 Split for classification
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    # 🎯 Classification Models
    print("\n🧪 Training classification models...")
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

        # Save model
        path = f"models/{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, path)
        print(f"💾 Saved {name} model to {path}")


    # 🔀 Split for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    print("\n📈 Training regression models...")

    # Linear Regression
    lin_model = train_linear_regression(X_train_reg, y_train_reg)
    lin_preds = lin_model.predict(X_test_reg)
    lin_mse = mean_squared_error(y_test_reg, lin_preds)
    print(f"Linear Regression MSE: {lin_mse:.2f}")
    joblib.dump(lin_model, 'models/linear_regression.joblib')
    print("💾 Saved Linear Regression model to models/linear_regression.joblib")


    # Polynomial Regression on numeric features only
    numeric_cols = ['tempo', 'length', 'loudness_(db)']
    X_train_poly = X_train_reg[numeric_cols]
    X_test_poly = X_test_reg[numeric_cols]

    poly_model = train_polynomial_regression(X_train_poly, y_train_reg)
    poly_preds = poly_model.predict(X_test_poly)
    poly_mse = mean_squared_error(y_test_reg, poly_preds)
    print(f"Polynomial Regression MSE: {poly_mse:.2f}")
    joblib.dump(poly_model, 'models/polynomial_regression.joblib')
    print("💾 Saved Polynomial Regression model to models/polynomial_regression.joblib")

    print("\n✅ Done.")

def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier()
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