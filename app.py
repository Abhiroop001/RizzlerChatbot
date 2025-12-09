import os
import json
import pickle
import threading
import logging
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

BASE_DIR = Path(__file__).resolve().parent

import nltk

NLTK_DATA_DIR = os.environ.get("NLTK_DATA", str(BASE_DIR / "nltk_data"))
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

nltk_resources = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]

for res in nltk_resources:
    try:
        if res == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif res == "punkt_tab":
            nltk.data.find("tokenizers/punkt_tab")
        else:
            nltk.data.find(f"corpora/{res}")
        logger.info("NLTK OK: %s", res)
    except LookupError:
        logger.info("Downloading NLTK: %s", res)
        nltk.download(res, download_dir=NLTK_DATA_DIR, quiet=False)
        if NLTK_DATA_DIR not in nltk.data.path:
            nltk.data.path.insert(0, NLTK_DATA_DIR)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from flask_cors import CORS

import tensorflow as tf

MODEL_PATH = BASE_DIR / "chatbot_model.h5"
WORDS_PATH = BASE_DIR / "words.pkl"
CLASSES_PATH = BASE_DIR / "classes.pkl"
INTENTS_PATH = BASE_DIR / "intents.json"

model = None
words = None
classes = None
intents = None
_model_lock = threading.Lock()

IGNORE_CHARS = {"?", "!", ".", ","}

app = Flask(__name__, static_folder=str(BASE_DIR / "static"), template_folder=str(BASE_DIR / "templates"))
frontend_origin = os.environ.get("FRONTEND_URL")
if frontend_origin:
    CORS(app, origins=[frontend_origin])
else:
    CORS(app)

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response

def load_artifacts():
    global model, words, classes, intents
    with _model_lock:
        if model is not None:
            return
        if not MODEL_PATH.exists() or not WORDS_PATH.exists() or not CLASSES_PATH.exists() or not INTENTS_PATH.exists():
            raise FileNotFoundError("Missing model or artifact files.")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        with open(WORDS_PATH, "rb") as f:
            words = pickle.load(f)
        with open(CLASSES_PATH, "rb") as f:
            classes = pickle.load(f)
        with open(INTENTS_PATH, encoding="utf-8") as f:
            intents = json.load(f)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words if w not in IGNORE_CHARS]

def bow(sentence, words_list):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words_list], dtype=np.float32)

def predict_class(sentence, thresh=0.25):
    load_artifacts()
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    out = [{"intent": classes[i], "probability": float(r)} for i, r in enumerate(res) if r > thresh]
    out.sort(key=lambda x: x["probability"], reverse=True)
    return out

def get_response(preds):
    load_artifacts()
    if not preds:
        return {"reply": "Sorry, I didn't understand that.", "intent": None, "confidence": 0.0}
    tag = preds[0]["intent"]
    conf = preds[0]["probability"]
    for i in intents["intents"]:
        if i["tag"] == tag:
            return {"reply": np.random.choice(i["responses"]), "intent": tag, "confidence": conf}
    return {"reply": "Sorry, I didn't understand that.", "intent": tag, "confidence": conf}

@app.route("/")
def ui():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    message = data.get("message") or data.get("text") or ""
    if not message.strip():
        return jsonify({"error": "Empty message"}), 400
    try:
        preds = predict_class(message)
        resp = get_response(preds)
        return jsonify(resp)
    except Exception as e:
        tb = traceback.format_exc()
        if os.environ.get("FLASK_DEBUG"):
            return jsonify({"error": str(e), "traceback": tb}), 500
        return jsonify({"error": str(e)}), 500

@app.route("/static/<path:filename>")
def static_files(filename):
    static_dir = BASE_DIR / "static"
    file = static_dir / filename
    if not file.exists():
        abort(404)
    return send_from_directory(str(static_dir), filename)

@app.errorhandler(404)
def not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = bool(os.environ.get("FLASK_DEBUG", False))
    app.run(host="0.0.0.0", port=port, debug=debug)
