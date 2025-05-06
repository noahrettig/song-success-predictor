from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def scale_numerical_features(X, numerical_cols):
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    return X

def encode_categorical_features(X, categorical_cols):
    X = pd.get_dummies(X, columns=categorical_cols)
    return X
