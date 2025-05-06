from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_text, max_features=500):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = tfidf.fit_transform(X_text)
    return X_tfidf, tfidf
