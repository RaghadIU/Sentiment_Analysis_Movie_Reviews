import os
import pandas as pd

def load_reviews(folder_path, label):
    reviews = []
    labels = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            reviews.append(file.read())
            labels.append(label)

    return reviews, labels


def load_imdb_dataset(base_path):
    train_pos = os.path.join(base_path, 'train', 'pos')
    train_neg = os.path.join(base_path, 'train', 'neg')
    test_pos = os.path.join(base_path, 'test', 'pos')
    test_neg = os.path.join(base_path, 'test', 'neg')

    X_train_pos, y_train_pos = load_reviews(train_pos, 1)
    X_train_neg, y_train_neg = load_reviews(train_neg, 0)

    X_test_pos, y_test_pos = load_reviews(test_pos, 1)
    X_test_neg, y_test_neg = load_reviews(test_neg, 0)

    train_df = pd.DataFrame({
        'review': X_train_pos + X_train_neg,
        'sentiment': y_train_pos + y_train_neg
    })

    test_df = pd.DataFrame({
        'review': X_test_pos + X_test_neg,
        'sentiment': y_test_pos + y_test_neg
    })

    return train_df, test_df


if __name__ == "__main__":
    train, test = load_imdb_dataset("data/raw/aclImdb")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
