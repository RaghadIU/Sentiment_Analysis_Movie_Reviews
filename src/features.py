import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features():
    train_df = pd.read_csv("data/processed/train_clean.csv")
    test_df = pd.read_csv("data/processed/test_clean.csv")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_df['clean_review'])
    X_test = vectorizer.transform(test_df['clean_review'])

    y_train = train_df['sentiment']
    y_test = test_df['sentiment']

    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = extract_features()
    print("TF-IDF feature extraction completed.")
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)
