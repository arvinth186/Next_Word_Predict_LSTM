import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the trained model
model = load_model('next_word_prediction_model_lstm.keras')

# Load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)


# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([text])[0]
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length-1, padding='pre')
    predicted_probs = model.predict(padded_sequence, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

# Streamlit app

st.title("Next Word Prediction using LSTM")
input_text = st.text_input("Enter a sequence of words:")
if st.button("Predict Next Word"):
    if input_text:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f"Predicted Next Word: **{next_word}**")
    else:
        st.write("Please enter some text to predict the next word.")
 