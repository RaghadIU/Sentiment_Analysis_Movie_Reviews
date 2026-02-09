import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from features import extract_features


def evaluate_model():
    X_train, X_test, y_train, y_test = extract_features()

    model = joblib.load("models/naive_bayes_model.pkl")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with open("results/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot()
    plt.title("Confusion Matrix - Naive Bayes")
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    print("Model evaluation completed.")
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    evaluate_model()
