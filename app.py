# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:00:57 2025

@author: grade
"""

import streamlit as st
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
print(os.listdir('.')) 

# Load model and tokenizer
model = keras.models.load_model("model.keras")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

max_length = 872

def predict_sentiment(text):
    text = text.lower()
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)[0]
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[predicted_label]

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment.")

user_input = st.text_area("Enter your text:")

if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text.")
