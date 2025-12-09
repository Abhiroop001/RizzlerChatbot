# application.py
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from tensorflow.keras.models import load_model
import os
import sys

# ---------------------------
# NLTK: ensure data is present
# ---------------------------
# Put nltk_data next to this file so it's shipped with your app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# make sure nltk looks here first
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# Required packages
_required_nltk = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]
for pkg in _required_nltk:
    try:
        # some resources live under tokenizers/..., some under corpora/...
        if pkg in ("punkt", "punkt_tab"):
            nltk.data.find(f"tokenizers/{pkg}")
        else:
            nltk.data.find(pkg)
    except LookupError:
        # attempt download to the local nltk_data
        try:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR)
        except Exception as e:
            # log but continue: we'll handle runtime lookup errors later
            print(f"Warning: failed to download NLTK package {pkg}: {e}", file=sys.stderr)

# ---------------------------
# Model & assets loading
# ---------------------------
lemmatizer = WordNetLemmatizer()

def safe_load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle {path}: {e}", file=sys.stderr)
        return None

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading json {path}: {e}", file=sys.stderr)
        return None

def safe_load_model(path):
    try:
        return load_model(path)
    except Exception as e:
        print(f"Error loading keras model {path}: {e}", file=sys.stderr)
        return None

# Resolve file paths relative to this file
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
WORDS_PKL = os.path.join(BASE_DIR, "words.pkl")
CLASSES_PKL = os.path.join(BASE_DIR, "classes.pkl")
MODEL_H5 = os.path.join(BASE_DIR, "chatbot_model.h5")

intents = safe_load_json(INTENTS_PATH)
words = safe_load_pickle(WORDS_PKL)
classes = safe_load_pickle(CLASSES_PKL)
model = safe_load_model(MODEL_H5)

# Print quick health info at startup to logs
print(f"NLTK data path: {nltk.data.path}", file=sys.stderr)
print(f"intents loaded: {bool(intents)}", file=sys.stderr)
print(f"words loaded: {len(words) if words else 0}", file=sys.stderr)
print(f"classes loaded: {len(classes) if classes else 0}", file=sys.stderr)
print(f"model loaded: {bool(model)}", file=sys.stderr)

# ---------------------------
# Flask app
# ---------------------------
application = Flask(__name__)
# also expose `app` for gunicorn commands expecting application:app
app = application

# ---------------------------
# CORS configuration
# ---------------------------
# Restrict origins in production. Add any allowed frontends here.
ALLOWED_ORIGINS = {
    "https://rizzlerchatbot-cl20.onrender.com",
    "https://rizzlerchatbot.onrender.com",
    # add other allowed origins if needed
}

# Use flask-cors to cover the common cases (registers OPTIONS handlers automatically).
# This settings allows CORS for routes matching /api/* to the specified origins.
CORS(
    application,
    resources={r"/api/*": {"origins": list(ALLOWED_ORIGINS)}},
    supports_credentials=True,
    expose_headers=["Content-Type"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# A final safety-net to ensure CORS headers are present even on error responses.
# It will echo back the Origin header only if it's in our allowed list.
@application.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    # ensure these are present for preflight and XHR cases
    response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
    response.headers.setdefault("Access-Control-Allow-Credentials", "true")
    return response

# Explicit OPTIONS handler for the /api/chat endpoint (helps some proxies)
@application.route("/api/chat", methods=["OPTIONS"])
def chat_options():
    response = make_response("", 204)
    origin = request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# ---------------------------
# NLP helper functions
# ---------------------------
def clean_up_sentence(sentence: str):
    # ensure sentence is a str
    if sentence is None:
        return []
    sentence = str(sentence)
    # guard in case punkt is still missing at runtime
    try:
        sentence_words = nltk.word_tokenize(sentence)
    except LookupError:
        # try to download punkt at runtime (best-effort)
        try:
            nltk.download("punkt", download_dir=NLTK_DATA_DIR)
            sentence_words = nltk.word_tokenize(sentence)
        except Exception:
            # fallback: very naive split
            sentence_words = sentence.split()
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence: str) -> np.ndarray:
    if not words:
        return np.array([], dtype=np.float32)
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag, dtype=np.float32)

def predict_class(sentence: str, error_threshold: float = 0.25):
    if model is None or words is None or classes is None:
        # no model loaded — return empty list so the caller can handle it
        return []
    bow = bag_of_words(sentence)
    if bow.size == 0:
        return []
    try:
        res = model.predict(np.array([bow]), verbose=0)[0]
    except Exception as e:
        print(f"Model prediction error: {e}", file=sys.stderr)
        return []
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": float(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understood that. Could you rephrase?", "unknown", 0.0
    top = intents_list[0]
    tag = top["intent"]
    prob = top["probability"]
    for intent in intents_json.get("intents", []):
        if intent.get("tag") == tag:
            return random.choice(intent.get("responses", ["Sorry, I don't know what to say."])), tag, prob
    return "Sorry, I couldn't find a good answer.", "unknown", prob

# ---------------------------
# Routes
# ---------------------------
@application.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Chatbot API is running", "health": "ok"})

@application.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(model),
        "intents_loaded": bool(intents),
    })

@application.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = data.get("message", "")
    message = (message or "").strip()
    if not message:
        return jsonify({
            "reply": "Please type something so I can respond 😊",
            "intent": "empty_message",
            "confidence": 0.0,
        }), 200

    # If model missing, return friendly error
    if model is None or words is None or classes is None or intents is None:
        return jsonify({
            "reply": "Service temporarily unavailable (model not loaded).",
            "intent": "service_unavailable",
            "confidence": 0.0,
        }), 503

    intents_list = predict_class(message)
    reply, intent, prob = get_response(intents_list, intents)
    return jsonify({
        "reply": reply,
        "intent": intent,
        "confidence": prob,
    }), 200

# Optional local run block (ignored by Gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", port=port, debug=True)
