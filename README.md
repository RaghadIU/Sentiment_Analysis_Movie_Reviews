# Sentiment Analysis on Movie Reviews 🎬

## Project Overview
This project is developed as part of a **Natural Language Processing (NLP)** course.  
The objective of the project is to build an NLP system that analyzes movie reviews and
classifies their overall sentiment as **Positive** or **Negative**.

With the increasing influence of online platforms and social media, movie reviews have
become an essential source of feedback for movie studios. This system helps automatically
analyze public opinion using machine learning techniques.

---

## Objectives
The main goals of this project are:
- Collect and prepare movie review data
- Apply text preprocessing techniques
- Train a supervised machine learning model
- Evaluate the model’s performance on unseen data
- Predict sentiment for new movie reviews
- Provide a simple web-based user interface for interaction

---

## Dataset
The project uses the **Stanford Large Movie Review Dataset (IMDb)**, which contains:
- 50,000 movie reviews
- Balanced positive and negative labels
- Separate training and testing sets

Dataset structure:
- 25,000 training reviews
- 25,000 testing reviews

---


---

## Methodology

### 1. Data Collection
Movie reviews are collected from the IMDb dataset, which is publicly available and
widely used in sentiment analysis research.

### 2. Data Preprocessing
The following preprocessing steps are applied:
- Lowercasing text
- Removing punctuation and special characters
- Removing stop words
- Tokenization
- Text vectorization using **TF-IDF**

### 3. Model Training
A **Naive Bayes** classifier is trained using supervised learning on labeled data
(positive and negative reviews).

### 4. Model Evaluation
The trained model is evaluated on a separate test dataset using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score

### 5. Sentiment Prediction
The system predicts the sentiment of new, unseen movie reviews as either:
- Positive
- Negative

---

## Web Application
A simple and interactive web interface is built using **Streamlit**.
Users can:
- Enter a movie review
- Click a button to analyze sentiment
- Instantly view the prediction result

---

## How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the Web Application
```bash
streamlit run app.py
```

## Example Usage
### Input 
```bash
The movie had an excellent storyline and great acting.
```
### Output 

Positive Sentiment

## Technologies Used

- Python

- Natural Language Processing (NLP)

- Scikit-learn

- NLTK

- Streamlit

- Joblib

## Conclusion

This project demonstrates the complete pipeline of an NLP-based sentiment analysis system,
from data collection and preprocessing to model training, evaluation, and deployment through
a web interface. The system effectively classifies movie reviews and can be extended to
support more advanced models or additional sentiment categories in the future.