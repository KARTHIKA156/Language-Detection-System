Language-Detection-System

Overview
      This project is a Machine Learning based Language Identification System that can automatically detect the language of a given text. It uses Natural Language Processing (NLP) techniques like TF-IDF vectorization for feature extraction and a Multinomial Naive Bayes classifier for training and prediction.

The project includes two main workflows:
Model Training & Saving – Reads dataset, cleans text, trains the model, evaluates performance, and saves it as .pkl files.
Language Prediction – Loads the trained model and instantly predicts the language of any input text.

 Project Structure
├── language_dataset_500.csv      # Dataset (sample: text + language label)
├── train_model.py                # First program: model training & saving
├── predict_language.py           # Second program: load model & predict
├── lang_model.pkl                # Saved trained model
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── label_encoder.pkl             # Saved label encoder
└── README.md                     # Project documentation

Features

Supports multiple languages (English, French, German, Hindi).
Preprocessing: Cleans text, removes unwanted symbols, and lowercases.
Feature Extraction: Uses TF-IDF Vectorizer.
Model: Multinomial Naive Bayes Classifier.
Evaluation: Provides Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
Instant Prediction: Load saved model to predict language without retraining.

Installation & Requirements
Make sure you have Python 3.x installed. Install the required libraries:
pip install pandas numpy scikit-learn seaborn matplotlib

Dataset
The dataset (language_dataset_500.csv) contains two columns:
text → Sample sentence
language → Corresponding language label

Example:
text	language
How are you?	English
Comment allez-vous ?	French
नमस्ते, आप कैसे हैं	Hindi
Das ist ein schönes Haus	German
 
How to Run
Train the Model
python train_model.py

This will:
Clean and preprocess text.
Train the Naive Bayes classifier.
Display evaluation metrics + confusion matrix.
Save trained model and vectorizers (.pkl files).

Predict Language
python predict_language.py


This will:
Load the saved model & vectorizer.
Predict the language of input texts.

Example Output:
Text: Comment allez-vous
Predicted Language: French
------------------------------
Text: Ich bin krank
Predicted Language: German
------------------------------
Text: नमस्ते, आप कैसे हैं
Predicted Language: Hindi
------------------------------

Model Performance
Accuracy: ~95% (depending on dataset split)
High precision & recall across all supported languages.
Confusion Matrix visualization for better insights.

 Conclusion
This project demonstrates how Natural Language Processing + Machine Learning can be combined to build a robust Language Identification System. It is efficient, scalable, and can be extended to more languages by expanding the dataset.

