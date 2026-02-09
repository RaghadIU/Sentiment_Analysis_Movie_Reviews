import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from features import extract_features


def train_naive_bayes():
    X_train, X_test, y_train, y_test = extract_features()

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    joblib.dump(model, "models/naive_bayes_model.pkl")

    print("Naive Bayes model trained successfully.")
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)


if __name__ == "__main__":
    train_naive_bayes()
