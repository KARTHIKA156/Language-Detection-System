Language Detection System

Overview
A Machine Learning + NLP project to automatically detect the language of a given text. Supports English, French, German, and Hindi.

Project Structure
├── language_dataset_500.csv   # Dataset
├── train_model.py             # Training & saving model
├── predict_language.py        # Predict using saved model
├── lang_model.pkl             # Trained Naive Bayes model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── label_encoder.pkl          # Label encoder
└── README.md                  # Documentation

Features
Text preprocessing & cleaning
TF-IDF vectorization
Multinomial Naive Bayes classifier
Accuracy, Precision, Recall, F1-score
Confusion Matrix visualization
Instant prediction without retraining

Installation
pip install pandas numpy scikit-learn seaborn matplotlib

Dataset Example
text	language
How are you?	English
Comment allez-vous ?	French
नमस्ते, आप कैसे हैं	Hindi
Das ist ein schönes Haus	German

Usage
Train the Model
python train_model.py

Predict Language
python predict_language.py

Example Output:
Text: Comment allez-vous
Predicted Language: French
------------------------------
Text: नमस्ते, आप कैसे हैं
Predicted Language: Hindi

Results
Accuracy: ~95%
Works well for short & long sentences
Easily extendable to more languages

Conclusion
A simple yet effective Language Detection System using NLP + ML.
Can be applied in chatbots, translators, and multilingual AI tools.
