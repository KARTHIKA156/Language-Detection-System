import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")
languages = pd.read_csv(r'C:\Users\Admin\Desktop\Language detection\Language Detection\language_dataset_500.csv')
text = languages['text']
lang  = languages['language']
data_list = []
for texts in text:
    texts = re.sub(r'[!@#$(),"%^*?:;~`0-9]', ' ', texts)
    texts = re.sub(r'\[|\]', ' ', texts)
    texts = texts.lower()
    data_list.append(texts)
tf = TfidfVectorizer()
X = tf.fit_transform(data_list).toarray()
le = LabelEncoder()
y = le.fit_transform(lang)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
langIden_model = MultinomialNB()
langIden_model.fit(X_train, y_train)
y_pred = langIden_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
conf_matrix = confusion_matrix(y_test, y_pred)
labels_present = np.unique(y_test)
class_report = classification_report(
    y_test, y_pred,
    labels=labels_present,
    target_names=le.inverse_transform(labels_present)
)
print("\nClassification Report:")
print(class_report)
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
with open("lang_model.pkl", "wb") as f:
    pickle.dump(langIden_model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tf, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Model and files saved successfully!")
