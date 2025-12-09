# app.py
import os
import json
import pickle
import threading
import logging
from pathlib import Path

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from flask_cors import CORS
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

# Ensure NLTK resources (same as before)
nltk_resources = ["punkt", "wordnet", "omw-1.4"]
for res in nltk_resources:
    try:
        if res == "punkt":
            nltk.data.find("tokenizers/punkt")
        else:
            nltk.data.find(f"corpora/{res}")
    except LookupError:
        logger.info("Downloading NLTK resource: %s", res)
        nltk.download(res)

lemmatizer = WordNetLemmatizer()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "chatbot_model.h5"
WORDS_PATH = BASE_DIR / "words.pkl"
CLASSES_PATH = BASE_DIR / "classes.pkl"
INTENTS_PATH = BASE_DIR / "intents.json"

model = None
words = None
classes = None
intents = None
_model_lock = threading.Lock()

# Flask app: ensure static_folder and template_folder point to the correct paths
app = Flask(__name__, static_folder=str(BASE_DIR / "static"), template_folder=str(BASE_DIR / "templates"))
frontend_origin = os.environ.get("FRONTEND_URL")
if frontend_origin:
    CORS(app, origins=[frontend_origin])
else:
    CORS(app)

def load_artifacts():
    global model, words, classes, intents
    with _model_lock:
        if model is not None and words is not None and classes is not None and intents is not None:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file at {MODEL_PATH}")
        if not WORDS_PATH.exists():
            raise FileNotFoundError(f"Missing words.pkl at {WORDS_PATH}")
        if not CLASSES_PATH.exists():
            raise FileNotFoundError(f"Missing classes.pkl at {CLASSES_PATH}")
        if not INTENTS_PATH.exists():
            raise FileNotFoundError(f"Missing intents.json at {INTENTS_PATH}")

        logger.info("Loading model...")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        with open(WORDS_PATH, "rb") as f:
            words = pickle.load(f)
        with open(CLASSES_PATH, "rb") as f:
            classes = pickle.load(f)
        with open(INTENTS_PATH, encoding="utf-8") as f:
            intents = json.load(f)
        logger.info("Artifacts loaded.")

IGNORE_CHARS = {"?", "!", ".", ","}

def clean_up_sentence(sentence: str):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words if w not in IGNORE_CHARS]

def bow(sentence: str, words_list):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words_list]
    return np.array(bag, dtype=np.float32)

def predict_class(sentence: str, thresh: float = 0.25):
    load_artifacts()
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    results = []
    for i, r in enumerate(res):
        if r > thresh:
            results.append({"intent": classes[i], "probability": float(r)})
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

def get_response(predicted_intents):
    load_artifacts()
    if not predicted_intents:
        return {"reply": "Sorry, I didn't understand that.", "intent": None, "confidence": 0.0}
    tag = predicted_intents[0]["intent"]
    confidence = predicted_intents[0]["probability"]
    for i in intents.get("intents", []):
        if i.get("tag") == tag:
            responses = i.get("responses", [])
            if responses:
                return {"reply": np.random.choice(responses), "intent": tag, "confidence": confidence}
    return {"reply": "Sorry, I didn't understand that.", "intent": tag, "confidence": confidence}

@app.route("/", methods=["GET"])
def ui():
    # Serve the UI. Replace with index.html in templates folder.
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400
    message = data.get("message") or data.get("text") or ""
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please provide non-empty 'message' in JSON body."}), 400
    try:
        preds = predict_class(message)
        resp = get_response(preds)
        return jsonify(resp)
    except FileNotFoundError as fnf:
        logger.exception("Missing artifact")
        return jsonify({"error": "Server misconfiguration: missing model files", "detail": str(fnf)}), 500
    except Exception as e:
        logger.exception("Unhandled exception")
        return jsonify({"error": "Server error", "detail": str(e)}), 500

# Optional: explicit static file route (Flask usually handles this)
@app.route("/static/<path:filename>")
def static_files(filename):
    static_dir = BASE_DIR / "static"
    target = static_dir / filename
    if not target.exists():
        # Return 404 JSON for API-like requests, or abort to let browser see 404 HTML.
        abort(404)
    return send_from_directory(str(static_dir), filename)

# Helpful 404 for debugging (keeps static 404 as real 404)
@app.errorhandler(404)
def handle_404(e):
    # Return the default 404 page (HTML) for browser navigation; APIs will still get 404.
    return "Not Found", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = bool(os.environ.get("FLASK_DEBUG", False))
    logger.info("Starting Flask app on 0.0.0.0:%s (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
