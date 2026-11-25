"""Feature engineering: Bag-of-Words and TF-IDF (unigram / bigram) vectorizers."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_vectorizers():
    """Return a fresh dict {name: vectorizer}. Input is already-preprocessed text."""
    return {
        "BoW (unigram)": CountVectorizer(min_df=2, max_features=20000),
        "TF-IDF (unigram)": TfidfVectorizer(min_df=2, max_features=20000, sublinear_tf=True),
        "TF-IDF (1-2 gram)": TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True),
    }
