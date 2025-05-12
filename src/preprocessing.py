from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def main():
    print("Loading dataset...")
    df = pd.read_csv('data/spotify_dataset.csv')
    print("Dataset loaded.")

    # Ignore "sus" attributes
    trusted_columns = [
        'text', 'Length', 'emotion', 'Genre',
        'Release Date', 'Key', 'Tempo', 'Loudness (db)',
        'Explicit', 'Popularity'
    ]
    df = df[trusted_columns]

    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    print("Preprocessing data...")
    # These are apparently the standard thresholds for Spotify popularity analysis 
    bins = [0, 25, 50, 70, 85, 100]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    df['success_level'] = pd.cut(df['popularity'], bins=bins, labels=labels, right=True)

    print(df.columns)

    # Convert mm:ss to seconds
    df['length'] = df['length'].apply(convert_to_seconds)

    # Clean emotion column
    valid_emotions = { "joy", "sadness", "anger", "fear", "love", "surprise" }
    # Setting invalid emotions to NaN
    df['emotion'] = df['emotion'].apply(lambda x: x.lower().strip() if isinstance(x, str) else x)
    df.loc[~df['emotion'].isin(valid_emotions), 'emotion'] = pd.NA

    # Drop rows with missing values
    df = df.dropna()

    # Convert release_date to release_month and release_year
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df['years_since_release'] = 2025 - df['release_year']
    # Log-transform years since release
    df['years_since_release_log'] = np.log1p(df['years_since_release'])
    df['release_month'] = df['release_date'].dt.month

    # Expand multi-genre column
    df = expand_genres(df, column="genre")

    # Change explicit to binary value
    df['explicit'] = df['explicit'].str.contains("Yes", case=False).astype(int)

    # One-hot encode remaining categorical attributes
    categorical_cols = ['emotion', 'key'] 
    df = encode_categorical_features(df, categorical_cols)

    # Normalize numerical attributes
    numerical_cols = ['length', 'tempo', 'loudness_(db)', 'years_since_release', 'years_since_release_log', 'release_month']
    df = scale_numerical_features(df, numerical_cols)

    # TF-IDF vectorization
    print("Vectorizing lyrics...")
        
    # df_sample = df.sample(9000, random_state=42).reset_index(drop=True) # TEMP: sample a smaller subset to reduce memory pressure
    df_sample = df

    tfidf_matrix, tfidf_model = tfidf_features(df_sample['text'], max_features=500)
    lyrics_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_model.get_feature_names_out())

    # Combine features
    drop_cols = ['release_date', 'text', 'popularity', 'success_level', 'time_signature']
    X_base = df_sample.drop(columns=[col for col in drop_cols if col in df_sample.columns]).reset_index(drop=True)
    X = pd.concat([X_base, lyrics_df.reset_index(drop=True)], axis=1)
    y = df_sample[['success_level']].copy()
    y.loc[:, 'popularity'] = df_sample['popularity']

    # Drop features that are highly collinear (corr > 0.95)
    X_numeric = X.select_dtypes(include=[np.number])
    corr_matrix = X_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]

    # Suggest whitelist features based on keywords
    keywords = ['pop', 'rock', 'rap', 'trap', 'hip', 'metal', 'love', 'know', 'like', 'tempo', 'loud', 'joy', 'sad', 'happy']
    whitelist = [col for col in high_corr_cols if any(k in col.lower() for k in keywords)]

    print(f"\nDropped {len(high_corr_cols)} highly correlated columns.")
    print("Whitelist-worthy suggestions (likely useful despite correlation):")
    for col in whitelist:
        print(f"  - {col}")

    # Drop only those NOT in whitelist
    to_drop = [col for col in high_corr_cols if col not in whitelist]
    X = X.drop(columns=to_drop)

    # log-scaling popularity
    df_sample['popularity_log'] = np.log1p(df_sample['popularity'])
    y = df_sample[['success_level']].copy()
    y.loc[:, 'popularity'] = df_sample['popularity_log']

    # Save results
    print("Saving processed data...")
    X.to_csv('data/X_processed.csv', index=False)
    y.to_csv('data/y_labels.csv', index=False)

    print("Preprocessing complete.")

# Helper functions
def scale_numerical_features(X, numerical_cols):
    scaler = MinMaxScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    return X

def encode_categorical_features(X, categorical_cols):
    X = pd.get_dummies(X, columns=categorical_cols)
    return X

def expand_genres(df, column='genre', delimiter=',', min_genre_count=10):
    from sklearn.preprocessing import MultiLabelBinarizer

    df[column] = df[column].fillna('')
    df['genre_list'] = df[column].apply(lambda x: [g.strip() for g in x.split(delimiter) if g.strip() != ''])

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genre_list'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=df.index)

    # Identify rare genres
    genre_counts = genre_df.sum()
    rare_genres = genre_counts[genre_counts < min_genre_count].index
    common_genres = genre_counts[genre_counts >= min_genre_count].index

    # Keep common genres, drop rare ones
    genre_common_df = genre_df[common_genres]

    # Create "Other" column for songs with only rare genres
    df['has_common_genre'] = genre_common_df.sum(axis=1) > 0
    genre_common_df['Other'] = ~df['has_common_genre']
    df = df.drop(columns=[column, 'genre_list', 'has_common_genre'])

    print(f"Kept {len(common_genres)} common genres. Assigned 'Other' to {genre_common_df['Other'].sum()} songs.")

    return pd.concat([df, genre_common_df], axis=1)

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
