import joblib
from preprocessing import clean_text


def predict_sentiment(review):
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/naive_bayes_model.pkl")

    clean_review = clean_text(review)
    vector = vectorizer.transform([clean_review])

    prediction = model.predict(vector)[0]

    return "Positive" if prediction == 1 else "Negative"


if __name__ == "__main__":
    review = input("Enter a movie review: ")
    result = predict_sentiment(review)
    print("Sentiment:", result)
