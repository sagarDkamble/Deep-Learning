import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model
model = load_model("imdb_rnn_model.keras")

# Load word index
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

vocab_size = 10000
maxlen = 200

def encode_review(text):
    text = text.lower().split()
    encoded = [word_index.get(w, 0) for w in text if word_index.get(w, 0) < vocab_size]
    padded = pad_sequences([encoded], maxlen=maxlen, padding="post", truncating="post")
    return padded

st.title("🎬 IMDB Sentiment Analyzer")

review = st.text_area("Enter movie review")

if st.button("Predict"):
    encoded = encode_review(review)
    prob = model.predict(encoded)[0][0]
    label = "POSITIVE 😊" if prob >= 0.5 else "NEGATIVE 😞"
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: `{prob:.2f}`")
