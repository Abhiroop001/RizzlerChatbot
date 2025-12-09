import os
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

INTENTS_FILE = "intents.json"
if not os.path.exists(INTENTS_FILE):
    raise FileNotFoundError(f"Could not find '{INTENTS_FILE}' in {os.path.abspath(os.getcwd())}.")

with open(INTENTS_FILE, encoding="utf-8") as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_chars = ["?", "!", ".", ","]

for intent in intents.get("intents", []):
    tag = intent.get("tag")
    for pattern in intent.get("patterns", []):
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, tag))
        if tag not in classes:
            classes.append(tag)

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

training_x = []
training_y = []
output_empty = [0] * len(classes)

for doc in documents:
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training_x.append(bag)
    training_y.append(output_row)

training_x = np.array(training_x, dtype=np.float32)
training_y = np.array(training_y, dtype=np.float32)

indices = np.arange(len(training_x))
np.random.shuffle(indices)
training_x = training_x[indices]
training_y = training_y[indices]

trainX = training_x
trainY = training_y

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation="softmax")
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

model.save("chatbot_model.h5")

with open("history.pkl", "wb") as f:
    pickle.dump(hist.history, f)

print("Done")