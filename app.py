import streamlit as st
from tensorflow.keras.models import load_model
from gensim.models import word2vec
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras import preprocessing
from sklearn.preprocessing import LabelEncoder
from function.embedding import PatchEmbedding, MeanVectorizer
from function.text_generator import TextGenerator

# import h5 model and corpus
with open('./model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('./model/sequences_digit.pkl', 'rb') as f:
    sequences_digit = pickle.load(f)

model = load_model('./model/51_acc_language_model.h5')
classify_model = load_model('./model/text_classify_model.h5')
word_model = word2vec.Word2Vec.load('./model/word.model')
news = pd.read_csv('./data/vnexpress/csv/cleaned_vnexpress.csv', encoding='utf-8').drop(columns=['Unnamed: 0'])
generator = TextGenerator(model=model, tokenizer=tokenizer)

# functions
def create_input_gensim(data):
    sequences = data['tag'].to_list()
    input_gensim = []
    for sen in sequences:
        input_gensim.append(sen.split())
    return input_gensim

def normalize_text(text, tokenizer, max_length):
    tokeninzed_sequences = generator.normalize_text(text)
    sequences_digit = tokenizer.texts_to_sequences([tokeninzed_sequences])
    sequences_digit_padding = np.array(preprocessing.sequence.pad_sequences(sequences_digit, maxlen=max_length, padding='pre'))
    return sequences_digit_padding

def cluster_task(task, news, text):
    if task == "**Classify text**":
        temp = normalize_text(text=text, tokenizer=tokenizer, max_length=12731)
        prediction = classify_model.predict(temp)
        
        label_encoder = LabelEncoder()
        label_encoder.fit(news['topic'])
        topic_class = label_encoder.inverse_transform([np.argmax(prediction[0])])
        # st.text(topic_class)
        return topic_class

    elif task == "**Search similar papers**":
        input_gensim = create_input_gensim(news)
        pemb = PatchEmbedding(word_model=word_model, stopword_path="./data/vietnamese-stopwords-dash.txt")
        mvectorize = MeanVectorizer(word_model=word_model)
        
        question_embeddings = pemb.sentence_embedding(text)
        post_embeddings = pemb.post_embedding(input_gensim, length=len(input_gensim))
        # question_embeddings, post_embeddings = ts.embedding(word_model, text, input_gensim)

        mean_sentence_embedding = mvectorize.mean_vector_embedding(question_embeddings)
        mean_post_embedding = mvectorize.mean_posts_embedding(post_embeddings)
        
        mean_sentence_embedding = mvectorize.mean_vector_embedding(question_embeddings)
        mean_post_embedding = mvectorize.mean_posts_embedding(post_embeddings)
        similarity_score = mvectorize.text_cosine_similarity(mean_sentence_embedding, mean_post_embedding)
        similar_news = mvectorize.find_similarity(similarity_score, news)
        # st.dataframe(similar_news)
        return similar_news

# global variables
cluster_result, task = None, ""

# GUI
st.set_page_config(layout="wide")
st.title("Text generator and classification")

context = st.text_input("Context sentence", placeholder="Việt Nam")
n_words = st.number_input("Lenght of sentence", value=None, placeholder="Type a number...")
try:
    result = generator.generate_sentences(context, n_words)
    text = st.text_area("Generated text", result)
except Exception as ex:
    print(str(ex))

# Side bar
with st.sidebar:
    st.title("Select a task")
    
    task = st.radio(
        "What would you like to do?",
        ["**Classify text**", "**Search similar papers**"],
        captions = ["Laugh out loud.", "Get the popcorn.", "Never stop learning."])
    
    if st.button("Run task"):
        cluster_result = cluster_task(task, news, text)
        
st.header("Result")
st.write(cluster_result)
        