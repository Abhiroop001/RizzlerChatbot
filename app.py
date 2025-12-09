import os
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

nltk_resources = ["punkt", "wordnet", "omw-1.4"]
for res in nltk_resources:
    try:
        nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
    except LookupError:
        nltk.download(res)

lemmatizer = WordNetLemmatizer()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "chatbot_model.h5")
WORDS_PATH = os.path.join(os.path.dirname(__file__), "words.pkl")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "classes.pkl")
INTENTS_PATH = os.path.join(os.path.dirname(__file__), "intents.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model file at {MODEL_PATH}")
if not os.path.exists(WORDS_PATH):
    raise FileNotFoundError(f"Missing words.pkl at {WORDS_PATH}")
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Missing classes.pkl at {CLASSES_PATH}")
if not os.path.exists(INTENTS_PATH):
    raise FileNotFoundError(f"Missing intents.json at {INTENTS_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(WORDS_PATH, "rb") as f:
    words = pickle.load(f)

with open(CLASSES_PATH, "rb") as f:
    classes = pickle.load(f)

with open(INTENTS_PATH, encoding="utf-8") as f:
    intents = json.load(f)

ignore_chars = ["?", "!", ".", ","]

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words if w not in ignore_chars]
    return sentence_words

def bow(sentence, words_list):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words_list]
    return np.array(bag, dtype=np.float32)

def predict_class(sentence, model, words_list, classes_list, thresh=0.25):
    p = bow(sentence, words_list)
    res = model.predict(np.array([p]), verbose=0)[0]
    results = []
    for i, r in enumerate(res):
        if r > thresh:
            results.append({"intent": classes_list[i], "probability": float(r)})
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

def get_response(intents_json, predicted_intents):
    if not predicted_intents:
        return {"response": "I do not understand."}
    tag = predicted_intents[0]["intent"]
    for i in intents_json.get("intents", []):
        if i.get("tag") == tag:
            responses = i.get("responses", [])
            if responses:
                return {"response": np.random.choice(responses), "intent": tag}
    return {"response": "I do not understand."}

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Please provide 'message' in JSON body."}), 400
    preds = predict_class(message, model, words, classes)
    resp = get_response(intents, preds)
    return jsonify(resp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
