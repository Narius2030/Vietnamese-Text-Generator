import tensorflow as tf
from pyvi import ViTokenizer
import string
import numpy as np


class TextGenerator():
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
    
    def normalize_text(self, doc):
        doc = ViTokenizer.tokenize(doc) #Pyvi Vitokenizer library
        doc = doc.lower() #Lower
        tokens = doc.split() #Split in_to words
        table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
        tokens = [w.translate(table) for w in tokens]
        tokens = [word for word in tokens if word]
        return tokens

    def preprocess_input(self, doc):
        tokens = self.normalize_text(doc)
        tokens = self.tokenizer.texts_to_sequences(tokens)
        for digit in tokens:
            if not digit:
                raise Exception("Từ vựng không tồn tại trong kho")

        tokens = tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen=50, truncating='pre')
        return np.reshape(tokens, (1,50))

    def top_n_words(self, text_input, top_n=3):
        tokens = self.preprocess_input(text_input)
        predictions = self.model.predict(tokens)[0]
        # Lấy top k dự đoán cao nhất
        top_indices = np.argpartition(predictions, -top_n)[-top_n:]

        top_words = []
        for index in top_indices:
            for word, idx in self.tokenizer.word_index.items():
                if idx == index:
                    top_words.append(word)
                    break
        return top_words

    def generate_sentences(self, context, n_words):
        tokens = self.preprocess_input(context)
        for _ in range(n_words):
            next_digit = np.argmax(self.model.predict(tokens, verbose=0))
            tokens = np.append(tokens, next_digit)
            tokens = np.delete(tokens, 0)
            tokens = np.reshape(tokens, (1, 50))
        # Mapping to text
        tokens = np.reshape(tokens, (50))
        # print(tokens)
        out_word = []
        for token in tokens:
            for word, index in self.tokenizer.word_index.items():
                if index == token:
                    out_word.append(word)
                    break
        return ' '.join(out_word)
    
    def generate_possible_sentences(self, context:str, top_n:int, n_words:int):
        top_words = self.top_n_words(context, top_n=top_n)
        # Với mỗi từ trong top_words, tạo ra một câu và lưu vào danh sách generated_sentences
        generated_sentences = []
        for word in top_words:
            new_text_input = context + " " + word
            generated_sentence = self.generate_sentences(new_text_input, n_words=n_words)
            generated_sentences.append(generated_sentence)

        # In ra các câu đã được tạo ra
        for i, sentence in enumerate(generated_sentences, start=1):
            print("Generated sentence", i, ":", sentence)