import streamlit as st
import joblib

model = joblib.load("models/naive_bayes_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("Movie Review Sentiment Analysis")
st.markdown(
    """
    This web application analyzes a movie review and predicts  
    whether the overall sentiment is **Positive** or **Negative**.
    """
)

st.divider()

review = st.text_area(
    label="Enter your movie review:",
    placeholder="Example: The movie had a great storyline and excellent acting.",
    height=150
)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a movie review before submitting.")
    else:
        review_vectorized = vectorizer.transform([review])
        prediction = model.predict(review_vectorized)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")

st.divider()

st.caption("NLP Project – Sentiment Analysis on Movie Reviews")
