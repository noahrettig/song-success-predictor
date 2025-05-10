from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def main():
    print("🔄 Loading dataset...")
    df = pd.read_csv('data/spotify_dataset_labeled.csv')

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
    drop_cols = ['artist(s)', 'song', 'album', 'release_date', 'text', 'popularity', 'success_level']
    X_base = df_sample.drop(columns=[col for col in drop_cols if col in df_sample.columns]).reset_index(drop=True)
    X = pd.concat([X_base, lyrics_df.reset_index(drop=True)], axis=1)
    y = df_sample[['success_level']].copy()
    y.loc[:, 'popularity'] = df_sample['popularity']


    # Save results
    print("💾 Saving processed data...")
    X.to_csv('data/X_processed.csv', index=False)
    y.to_csv('data/y_labels.csv', index=False)
    print("✅ Preprocessing complete.")

def scale_numerical_features(X, numerical_cols):
    scaler = MinMaxScaler()
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

def tfidf_features(X_text, max_features=500):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = tfidf.fit_transform(X_text)
    return X_tfidf, tfidf

if __name__ == '__main__':
    main()
