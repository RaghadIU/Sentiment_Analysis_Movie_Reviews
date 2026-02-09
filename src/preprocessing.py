import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_df['clean_review'] = train_df['review'].apply(clean_text)
    test_df['clean_review'] = test_df['review'].apply(clean_text)

    train_df.to_csv("data/processed/train_clean.csv", index=False)
    test_df.to_csv("data/processed/test_clean.csv", index=False)

    print("Text preprocessing completed successfully.")
