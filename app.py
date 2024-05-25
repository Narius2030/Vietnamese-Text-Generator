import streamlit as st
from gensim.models import word2vec
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from function.generator import TextGenerator


# import h5 model and corpus
with open('./model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('./model/sequences_digit.pkl', 'rb') as f:
    sequences_digit = pickle.load(f)

model = load_model('./model/51_acc_language_model.h5')
word_model = word2vec.Word2Vec.load('./model/word.model')
news = pd.read_csv('./data/vnexpress/csv/cleaned_vnexpress.csv').drop(columns=['Unnamed: 0'])
generator = TextGenerator(model=model, tokenizer=tokenizer)

# functions
def create_input_gensim(data):
    sequences = data['tag'].to_list()
    input_gensim = []
    for sen in sequences:
        try:
            input_gensim.append(sen.split())
        except Exception as ex:
            pass
    return input_gensim

def normalize_text(text, tokenizer, max_length):
    tokeninzed_sequences = generator.normalize_text(text)
    sequences_digit = tokenizer.texts_to_sequences([tokeninzed_sequences])
    sequences_digit_padding = np.array(preprocessing.sequence.pad_sequences(sequences_digit, maxlen=max_length, padding='pre'))
    return sequences_digit_padding

# GUI
st.set_page_config(layout="wide")
st.title("Sporting Magazine Looker")

# Processing
context = st.text_input("Context sentence", placeholder="Việt Nam")
contexts = generator.generate_possible_sentences(context)
st.write(contexts)
try:
    result = generator.generate_sentences(context, 20)
    text = st.text_area("Generated text", result)
except Exception as ex:
    pass
        