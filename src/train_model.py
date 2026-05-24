import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from features import extract_features


def train_models():
    X_train, X_test, y_train, y_test = extract_features()

    # Naive Bayes Model
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    nb_pred = nb_model.predict(X_test)

    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_report = classification_report(y_test, nb_pred)

    joblib.dump(nb_model, "models/naive_bayes_model.pkl")

    print("\n===== Naive Bayes =====")
    print("Accuracy:", nb_accuracy)
    print(nb_report)

    # SVM Model
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)

    svm_pred = svm_model.predict(X_test)

    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_report = classification_report(y_test, svm_pred)

    joblib.dump(svm_model, "models/svm_model.pkl")

    print("\n===== SVM =====")
    print("Accuracy:", svm_accuracy)
    print(svm_report)


if __name__ == "__main__":
    train_models()