# app.py
import os
import json
import pickle
import threading
import logging
from pathlib import Path
import traceback

# ---------------------------
# Basic logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

# ---------------------------
# NLTK: ensure resources BEFORE anything else that tokenizes text
# ---------------------------
import nltk

# Destination for NLTK data (Render friendly). Respect env var if set.
NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "/opt/render/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Make sure nltk searches this directory first
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# Resources required by the app
_nltk_resources = ["punkt", "wordnet", "omw-1.4"]

for _res in _nltk_resources:
    try:
        if _res == "punkt":
            nltk.data.find("tokenizers/punkt")
        else:
            nltk.data.find(f"corpora/{_res}")
        logger.info("NLTK resource already present: %s", _res)
    except LookupError:
        logger.info("NLTK resource '%s' not found; downloading to %s", _res, NLTK_DATA_DIR)
        # download_dir ensures it goes into our chosen directory
        nltk.download(_res, download_dir=NLTK_DATA_DIR, quiet=False)
        # ensure path contains the download dir
        if NLTK_DATA_DIR not in nltk.data.path:
            nltk.data.path.insert(0, NLTK_DATA_DIR)
        logger.info("NLTK resource '%s' downloaded.", _res)

# Now safe to import lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Other imports (after nltk ready)
# ---------------------------
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from flask_cors import CORS

# Import TensorFlow after NLTK has been handled (keeps logs cleaner)
import tensorflow as tf

# ---------------------------
# Paths & Globals
# ---------------------------
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

IGNORE_CHARS = {"?", "!", ".", ","}

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, static_folder=str(BASE_DIR / "static"), template_folder=str(BASE_DIR / "templates"))
frontend_origin = os.environ.get("FRONTEND_URL")
if frontend_origin:
    CORS(app, origins=[frontend_origin])
    logger.info("CORS restricted to: %s", frontend_origin)
else:
    CORS(app)
    logger.info("CORS allowed for all origins (development)")

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response

# ---------------------------
# Model & artifacts loader (thread-safe)
# ---------------------------
def load_artifacts():
    """Load model, words, classes, intents. Thread-safe and idempotent."""
    global model, words, classes, intents
    with _model_lock:
        if model is not None and words is not None and classes is not None and intents is not None:
            return

        # Validate files
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file at {MODEL_PATH}")
        if not WORDS_PATH.exists():
            raise FileNotFoundError(f"Missing words.pkl at {WORDS_PATH}")
        if not CLASSES_PATH.exists():
            raise FileNotFoundError(f"Missing classes.pkl at {CLASSES_PATH}")
        if not INTENTS_PATH.exists():
            raise FileNotFoundError(f"Missing intents.json at {INTENTS_PATH}")

        logger.info("Loading TensorFlow model from %s", MODEL_PATH)
        # load_model can be heavy — do it while holding lock so only one thread loads it
        model = tf.keras.models.load_model(str(MODEL_PATH))
        logger.info("Model loaded.")

        logger.info("Loading words and classes pickles")
        with open(WORDS_PATH, "rb") as f:
            words = pickle.load(f)
        with open(CLASSES_PATH, "rb") as f:
            classes = pickle.load(f)

        logger.info("Loading intents json")
        with open(INTENTS_PATH, encoding="utf-8") as f:
            intents = json.load(f)

        logger.info("Artifacts loaded successfully.")

# ---------------------------
# Text processing helpers
# ---------------------------
def clean_up_sentence(sentence: str):
    """Tokenize and lemmatize input sentence."""
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

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET"])
def ui():
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
        logger.exception("Missing artifact while handling /api/chat")
        return jsonify({"error": "Server misconfiguration: missing model files", "detail": str(fnf)}), 500
    except Exception as e:
        # log full traceback
        tb = traceback.format_exc()
        logger.error("Unhandled exception in /api/chat:\n%s", tb)
        # Return traceback in JSON only if FLASK_DEBUG is set (developer)
        if os.environ.get("FLASK_DEBUG"):
            return jsonify({"error": "Server error", "detail": str(e), "traceback": tb}), 500
        return jsonify({"error": "Server error", "detail": str(e)}), 500

# Explicit static route (optional; Flask also serves static automatically)
@app.route("/static/<path:filename>")
def static_files(filename):
    static_dir = BASE_DIR / "static"
    target = static_dir / filename
    if not target.exists():
        abort(404)
    return send_from_directory(str(static_dir), filename)

# Keep default 404 helpful
@app.errorhandler(404)
def handle_404(e):
    return "Not Found", 404

# ---------------------------
# Run (for local dev only)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = bool(os.environ.get("FLASK_DEBUG", False))
    logger.info("Starting Flask app on 0.0.0.0:%s (debug=%s)", port, debug)
    # In production use gunicorn: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 4`
    app.run(host="0.0.0.0", port=port, debug=debug)