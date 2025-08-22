import re
import pickle
with open("lang_model.pkl", "rb") as f:
    langIden_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    tf = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
def predict(text):
    text_cleaned = re.sub(r'[!@#$(),"%^*?:;~`0-9]', ' ', text)
    text_cleaned = re.sub(r'\[|\]', ' ', text_cleaned)
    text_cleaned = text_cleaned.lower()
    x = tf.transform([text_cleaned])
    lang = langIden_model.predict(x)
    lang = le.inverse_transform(lang)
    print(f"Text: {text}")
    print("Predicted Language:", lang[0])
    print("-" * 30)
predict("Comment allez-vous")
predict("Ich bin krank")
predict("How are you today?")
predict("Paris est une belle ville")
predict("Das ist ein schönes Haus")
predict("नमस्ते, आप कैसे हैं")
predict("I love programming in Python")
predict("Wie geht es dir")
predict("Il fait très chaud aujourd'hui")
predict("आज मौसम बहुत अच्छा है")
predict("This is a beautiful morning")
predict("J’aime apprendre le français")
predict("Ich lerne gerne neue Sprachen")
predict("मुझे क्रिकेट खेलना पसंद है")
predict("My name is Teddy")


