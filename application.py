import json
import random
import pickle
import numpy as np
import nltk

from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab/english.pickle")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

with open("intents.json", encoding="utf-8") as f:
    intents = json.load(f)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

application = Flask(__name__)


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

    return_list = []
    for r in results:
        return_list.append(
            {
                "intent": classes[r[0]],
                "probability": float(r[1]),
            }
        )
    return return_list


def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understood that. Could you rephrase?"

    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]

    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    else:
        result = "Sorry, I couldn't find a good answer."

    return result


@application.route("/", methods=["GET"])
def home():
    return "Chatbot API is running."


@application.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    intents_list = predict_class(message)
    response_text = get_response(intents_list, intents)
    return jsonify({
        "message": message,
        "intents": intents_list,
        "response": response_text,
    })


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)
