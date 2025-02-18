import tkinter as tk
from tkinter import scrolledtext
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = keras.models.load_model("model.keras")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define the max_length used during training
max_length = 872  

def predict_sentiment():
    text = text_entry.get("1.0", tk.END).strip().lower()
    if not text:
        result_label.config(text="Please enter text to analyze.")
        return

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)[0]  
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    result_label.config(text=f"Predicted Sentiment: {label_map[predicted_label]}")

# Create GUI Window
root = tk.Tk()
root.title("Sentiment Analysis")

# Text Entry Box
text_entry = scrolledtext.ScrolledText(root, width=50, height=5)
text_entry.pack(pady=10)

# Predict Button
predict_button = tk.Button(root, text="Analyze Sentiment", command=predict_sentiment)
predict_button.pack(pady=5)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

# Run the GUI loop
root.mainloop()