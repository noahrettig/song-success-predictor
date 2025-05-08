import sys
import os

# Add the project root (one level up from this script) to the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

print("✅ Script loaded. Name is:", __name__) # Debug -- didn't run for me for some reason

import pandas as pd
from src.preprocessing import (
    scale_numerical_features,
    encode_categorical_features,
    convert_to_seconds
)
from src.feature_engineering import tfidf_features

def main():
    print("🔄 Loading dataset...")
    df = pd.read_csv('data/spotify_dataset_rendered_with_labels.csv')

    # Convert mm:ss to seconds
    df['length'] = df['length'].apply(convert_to_seconds)

    # Drop rows with essential missing values
    df = df.dropna(subset=['length', 'tempo', 'loudness_(db)', 'text', 'success_level'])

    # One-hot encode categorical attributes
    categorical_cols = ['genre', 'key', 'emotion', 'time_signature', 'explicit']
    df = encode_categorical_features(df, categorical_cols)

    # Normalize numerical attributes
    numerical_cols = ['length', 'tempo', 'loudness_(db)']
    df = scale_numerical_features(df, numerical_cols)

    # TF-IDF vectorization
    print("🧠 Vectorizing lyrics...")
        
    df_sample = df.sample(9000, random_state=42).reset_index(drop=True) # TEMP: sample a smaller subset to reduce memory pressure

    tfidf_matrix, tfidf_model = tfidf_features(df_sample['text'], max_features=500)
    lyrics_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_model.get_feature_names_out())

    # Combine features
    drop_cols = ['artist(s)', 'song', 'album', 'release_date', 'text', 'popularity', 'success_level', 'length']
    X_base = df_sample.drop(columns=[col for col in drop_cols if col in df_sample.columns]).reset_index(drop=True)
    X = pd.concat([X_base, lyrics_df.reset_index(drop=True)], axis=1)
    y = df_sample['success_level']

    # Save results
    print("💾 Saving processed data...")
    X.to_csv('data/X_processed.csv', index=False)
    y.to_csv('data/y_labels.csv', index=False)
    print("✅ Preprocessing complete.")

if __name__ == '__main__':
    main()
