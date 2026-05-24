import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from features import extract_features


def evaluate_model(model_path, model_name):
    X_train, X_test, y_train, y_test = extract_features()

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n===== {model_name} Results =====")
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(report)

    with open(f"results/{model_name}_metrics.txt", "w") as f:
        f.write(f"{model_name} Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")

    plt.savefig(f"results/{model_name}_confusion_matrix.png")

    plt.show()


if __name__ == "__main__":
    evaluate_model(
        "models/naive_bayes_model.pkl",
        "Naive_Bayes"
    )

    evaluate_model(
        "models/svm_model.pkl",
        "SVM"
    )