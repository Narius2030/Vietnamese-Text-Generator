import streamlit as st
import plugins.text_generator as tg
import tensorflow as tf
import pickle

# import h5 model and corpus
with open('./model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('./model/sequences_digit.pkl', 'rb') as f:
    sequences_digit = pickle.load(f)

model = tf.keras.models.load_model('./model/51_acc_language_model.h5')

# GUI
st.set_page_config(layout="wide")
st.header("Text generator and classification")

context = st.text_input("Context sentence", placeholder="Việt Nam")
result = tg.generate_sentences(context)

txt = st.text_area("Generated text", result)