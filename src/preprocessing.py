from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def scale_numerical_features(X, numerical_cols):
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    return X

def encode_categorical_features(X, categorical_cols):
    X = pd.get_dummies(X, columns=categorical_cols)
    return X

def convert_to_seconds(time_str):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except:
        return None
    
