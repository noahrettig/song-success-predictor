from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def main():
    print("Loading dataset...")
    df = pd.read_csv('data/spotify_dataset.csv')
    print("Dataset loaded.")

    # Ignore "sus" attributes
    trusted_columns = [
        'Artist(s)', 'song', 'text', 'Length', 'emotion', 'Genre', 'Album',
        'Release Date', 'Key', 'Tempo', 'Loudness (db)', 'Time signature',
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
    df['release_month'] = df['release_date'].dt.month

    # Expand multi-genre column
    df = expand_genres(df, column="genre")

    # One-hot encode remaining categorical attributes
    categorical_cols = ['key', 'emotion', 'time_signature', 'explicit']
    df = encode_categorical_features(df, categorical_cols)

    # Normalize numerical attributes
    numerical_cols = ['length', 'tempo', 'loudness_(db)', 'years_since_release', 'release_month']
    df = scale_numerical_features(df, numerical_cols)

    # TF-IDF vectorization
    print("Vectorizing lyrics...")
        
    df_sample = df.sample(9000, random_state=42).reset_index(drop=True) # TEMP: sample a smaller subset to reduce memory pressure

    tfidf_matrix, tfidf_model = tfidf_features(df_sample['text'], max_features=500)
    lyrics_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_model.get_feature_names_out())

    # Combine features
    drop_cols = ['release_date', 'text', 'popularity', 'success_level']
    X_base = df_sample.drop(columns=[col for col in drop_cols if col in df_sample.columns]).reset_index(drop=True)
    X = pd.concat([X_base, lyrics_df.reset_index(drop=True)], axis=1)
    y = df_sample[['success_level']].copy()
    y.loc[:, 'popularity'] = df_sample['popularity']

    X_no_text = df.drop(columns=[col for col in drop_cols if col in df.columns]).reset_index(drop=True)
    y_no_text = df[['success_level']].copy()
    y_no_text.loc[:, 'popularity'] = df['popularity']

    # Save results
    print("Saving processed data...")
    X.to_csv('data/X_processed.csv', index=False)
    y.to_csv('data/y_labels.csv', index=False)

    X_no_text.to_csv('data/X_processed_textless.csv')
    y_no_text.to_csv('data/y_processed_textless.csv')

    print("Preprocessing complete.")

# Helper functions
def scale_numerical_features(X, numerical_cols):
    scaler = MinMaxScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    return X

def encode_categorical_features(X, categorical_cols):
    X = pd.get_dummies(X, columns=categorical_cols)
    return X

def expand_genres(df, column='genre', delimiter=','):
    """
    Takes a DataFrame and expands a comma-separated genre column into multi-hot encoded features.
    """
    # Fill NaNs with empty string to avoid errors
    df[column] = df[column].fillna('')
    
    # Split genre strings into lists
    df['genre_list'] = df[column].apply(lambda x: [g.strip() for g in x.split(delimiter) if g.strip() != ''])
    
    # Create dummy/multi-hot encoding
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_df = pd.DataFrame(mlb.fit_transform(df['genre_list']), columns=mlb.classes_, index=df.index)

    print(f"Expanded to {len(genre_df.columns)} unique genre columns.")

    # Drop original columns
    df = df.drop(columns=[column, 'genre_list'])

    # Join expanded genres back in
    df = pd.concat([df, genre_df], axis=1)

    return df

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
