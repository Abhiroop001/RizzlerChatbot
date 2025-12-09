import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# Load intents and model assets
with open("intents.json", encoding="utf-8") as f:
    intents = json.load(f)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

# Elastic Beanstalk expects the Flask app object to be named `application`
application = Flask(__name__)
CORS(application)


def clean_up_sentence(sentence: str):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence: str) -> np.ndarray:
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag, dtype=np.float32)


def predict_class(sentence: str, error_threshold: float = 0.25):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [
        {"intent": classes[r[0]], "probability": float(r[1])}
        for r in results
    ]


def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understood that. Could you rephrase?", "unknown", 0.0

    top = intents_list[0]
    tag = top["intent"]
    prob = top["probability"]

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            result = random.choice(intent["responses"])
            return result, tag, prob

    return "Sorry, I couldn't find a good answer.", "unknown", prob


@application.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Chatbot API is running", "health": "ok"})


@application.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@application.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify(
            {
                "reply": "Please type something so I can respond 😊",
                "intent": "empty_message",
                "confidence": 0.0,
            }
        )

    intents_list = predict_class(message)
    reply, intent, prob = get_response(intents_list, intents)

    return jsonify(
        {
            "reply": reply,
            "intent": intent,
            "confidence": prob,
        }
    )


if __name__ == "__main__":
    # Local run
    application.run(host="0.0.0.0", port=5000, debug=True)
