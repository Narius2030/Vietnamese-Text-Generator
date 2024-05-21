import streamlit as st
import plugins.text_generator as tg
import plugins.embedding as ts
from tensorflow.keras.models import load_model
from gensim.models import word2vec
import pandas as pd
import pickle

# import h5 model and corpus
with open('./model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('./model/sequences_digit.pkl', 'rb') as f:
    sequences_digit = pickle.load(f)

model = load_model('./model/51_acc_language_model.h5')
word_model = word2vec.Word2Vec.load('./model/word.model')
news = pd.read_csv('./data/vnexpress/csv/vnexpress_01.csv').drop(columns=['Unnamed: 0'])

# functions
def create_input_gensim(data):
    sequences = data['tag'].to_list()
    input_gensim = []
    for sen in sequences:
        input_gensim.append(sen.split())
    return input_gensim

# GUI
st.set_page_config(layout="wide")
st.header("Text generator and classification")

context = st.text_input("Context sentence", placeholder="Việt Nam")
result = tg.generate_sentences(tokenizer, model, context, 20)

text = st.text_area("Generated text", result)

if st.button("Find similar texts"):
    input_gensim = create_input_gensim(news)
    question_embeddings, post_embeddings = ts.embedding(word_model, text, input_gensim)
    
    mean_sentence_embedding = ts.mean_vector_embedding(word_model, question_embeddings)
    mean_post_embedding = ts.mean_embedded_posts(word_model, post_embeddings)
    similarity_scores = ts.text_cosine_similarity(mean_sentence_embedding, mean_post_embedding)
    similarity_posts = ts.find_similarity(similarity_scores, news)
    st.dataframe(similarity_posts)