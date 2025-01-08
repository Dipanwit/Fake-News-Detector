# Fake News Detector

## Overview
The Fake News Detector is a machine learning-based project designed to identify and classify news articles as either "fake" or "real" based on their content. This project aims to combat the spread of misinformation by providing a reliable tool to assess the authenticity of news articles.

## Features
- **Data Preprocessing**: Cleans and preprocesses the text data, including removing stopwords, punctuation, and performing tokenization.
- **Vectorization**: Converts textual data into numerical representations using techniques like TF-IDF.
- **Machine Learning Models**: Implements multiple algorithms such as Logistic Regression, Naive Bayes, and Support Vector Machines (SVM).
- **Performance Evaluation**: Compares models based on accuracy, precision, recall, and F1-score to select the best-performing algorithm.
- **User-Friendly Interface**: A Flask-based web application for users to input news articles and receive predictions.

## Tech Stack
- **Programming Language**: Python
- **Machine Learning Libraries**: Scikit-learn, Pandas, NumPy
- **Natural Language Processing (NLP)**: NLTK, TF-IDF
- **Web Framework**: Flask
- **Frontend**: HTML, CSS, JavaScript

## How It Works
1. **Input**: Users input a news article or its headline.
2. **Preprocessing**: The text is cleaned, tokenized, and vectorized.
3. **Prediction**: The preprocessed data is fed into the trained machine learning model to predict whether the news is real or fake.
4. **Output**: The system returns the prediction result with a confidence score.

## Dataset
- The dataset used for this project was sourced from [Kaggle](https://www.kaggle.com/). It contains labeled news articles categorized as "real" or "fake."
- Preprocessing steps include:
  - Removing null values
  - Tokenization
  - Lowercasing text
  - Removing special characters and numbers

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Access the application in your browser at `http://127.0.0.1:5000`.

## Model Training
1. Load the dataset and preprocess the text.
2. Split the data into training and testing sets.
3. Train multiple machine learning models (e.g., Logistic Regression, Naive Bayes).
4. Evaluate the models using metrics like accuracy, precision, recall, and F1-score.
5. Save the best-performing model using `joblib` for deployment.

## Example Output
- **Input**: "The moon landing was staged in a Hollywood studio."
- **Output**: Fake News (Confidence: 87%)

## Project Structure
```
Fake-News-Detector/
├── app.py            # Flask application
├── templates/        # HTML templates
├── static/           # CSS and JavaScript files
├── models/           # Saved machine learning models
├── data/             # Dataset files
├── requirements.txt  # Dependencies
├── README.md         # Project documentation
```

## Future Enhancements
- Expand the dataset to improve model generalization.
- Implement deep learning models like LSTMs or BERT for improved accuracy.
- Add multilingual support to detect fake news in different languages.
- Deploy the application to a cloud platform like AWS or Heroku for public access.



