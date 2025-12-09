import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab/english.pickle")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

lemmatizer = WordNetLemmatizer()

with open("../intents.json", encoding="utf-8") as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_chars = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [
    lemmatizer.lemmatize(w.lower())
    for w in words
    if w not in ignore_chars
]
words = sorted(set(words))

classes = sorted(set(classes))

with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [
        lemmatizer.lemmatize(w.lower()) for w in pattern_words
    ]

    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training, dtype=np.float32)

trainX = training[:, : len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Dense(
        128, input_shape=(len(trainX[0]),), activation="relu"
    )
)
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))

sgd = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9, nesterov=True
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"],
)

hist = model.fit(
    trainX,
    trainY,
    epochs=200,
    batch_size=5,
    verbose=1,
)

model.save("chatbot_model.h5")

with open("history.pkl", "wb") as f:
    pickle.dump(hist.history, f)

print("Done")