import pandas as pd
import os
from tqdm import tqdm
from pyvi import ViTokenizer
import sys
sys.path.append("./src/dtprocess")
import cleandt


class NormalizeTexts():
    def __init__(self) -> None:
        pass
    
    def get_info(self, topic, processed_news):
        temp = processed_news[processed_news.topic == topic]
        return temp['article_id'].to_list(), temp['tag'].to_list()
    
    def load_data(self, path):
        ## Import data from raw folder to dataframe
        CRAWL_FOLDER = path
        vnexpress = []

        for filename in os.listdir(CRAWL_FOLDER):
            with open(f'{CRAWL_FOLDER}/{filename}', 'r') as file:
                news = file.readlines()
                vnexpress += cleandt.convert_dict(news, 'content')
        news = pd.DataFrame(columns=['content','url','topic', 'sub-topic', 'image', 'title','description'])
        
        for new in vnexpress:
            news.loc[len(news)] = pd.Series(new)
        news = news.reset_index().rename(columns={'index':'article_id'})
        return news
        
    def dump_files(self, processed_news, path):
        """Dump each tag to a text file
        """
        processed_news.to_csv(f'{path}/csv/cleaned_vnexpress.csv')
        
        PROCESSED_FOLDER = f'{path}/processed_news'
        for topic in tqdm(os.listdir(PROCESSED_FOLDER)):
            articles_ids, tags = self.get_info(topic, processed_news)
            # print(articles_ids)
            for id, tag in zip(articles_ids, tags):
                with open(f'{PROCESSED_FOLDER}/{topic}/{id}.txt', "w", encoding="utf-8") as file:
                    file.write(tag)

    def transform_texts(self, data):
        """Transform the raw data to usable text
        """
        ## Select necessary columns
        processed_news = data[['article_id','content','topic','sub-topic','title','description','url']]

        ## Find null values and processing
        processed_news.fillna('', inplace=True)
        ## Merge columns into a single `tag` column
        processed_news['tag'] = processed_news['content'] + processed_news['title'] + processed_news['description']
        processed_news = processed_news.drop(columns=['content','description'])

        ## Tokenize the Vietnamese words
        processed_news['tag'] = processed_news['tag'].apply(lambda x: x.lower())
        processed_news['tag'] = processed_news['tag'].apply(lambda x: cleandt.remove_punctuation(x))
        processed_news['tag'] = processed_news['tag'].apply(lambda x: ViTokenizer.tokenize(x))
        return processed_news