import pandas as pd
import os
from tqdm import tqdm
from data_scrape.dtprocess import cleandt
from pyvi import ViTokenizer

ROOT_PATH = '/mnt/d/Programming/Vietnamese-Text-Generator/'

def lower_case(x):
    try:
        x = x.lower()
    except Exception as ex:
        pass
    return x

def get_info(topic, processed_news):
    temp = processed_news[processed_news.topic == topic]
    return temp['article_id'].to_list(), temp['tag'].to_list()

def transform():
    """Transform the raw data to usable text
    """
    ## Import data from raw folder to dataframe
    CRAWL_FOLDER = f'{ROOT_PATH}/data/test/raw'
    vnexpress = []

    for filename in os.listdir(CRAWL_FOLDER):

        with open(f'{CRAWL_FOLDER}/{filename}', 'r') as file:
            news = file.readlines()
            vnexpress += cleandt.convert_dict(news, 'content')
            
    news = pd.DataFrame(columns=['content','url','topic', 'sub-topic', 'image', 'title','description'])

    for new in vnexpress:
        news.loc[len(news)] = pd.Series(new)
    news = news.reset_index().rename(columns={'index':'article_id'})

    ## Select necessary columns
    processed_news = news[['article_id','content','topic','sub-topic','title','description']]
    print(processed_news.shape)
    processed_news.head()

    raw_news = processed_news.copy()

    ## Lower characters
    for col in processed_news.select_dtypes(include='object').columns:
        processed_news[col] = processed_news[col].apply(lambda x: lower_case(x))

    ## Find null values and processing
    processed_news.fillna('', inplace=True)
    ## Merge columns into a single `tag` column
    processed_news['tag'] = processed_news['content'] + processed_news['title'] + processed_news['description'] + processed_news['topic'] + processed_news['sub-topic']
    processed_news = processed_news.drop(columns=['content','description','title'])

    ## Tokenize the Vietnamese words
    processed_news['tag'] = processed_news['tag'].apply(lambda x: ViTokenizer.tokenize(x))
    processed_news['tag'] = processed_news['tag'].apply(lambda x: cleandt.remove_stopword(x, f'{ROOT_PATH}/data/vietnamese-stopwords.txt'))
    
    return raw_news, processed_news

def dump(raw_news, processed_news):
    """Dump each tag to a text file
    """
    raw_news.to_csv(f'{ROOT_PATH}/data/test/csv/vnexpress.csv')
    
    PROCESSED_FOLDER = f'{ROOT_PATH}/data/test/processed'
    for topic in tqdm(os.listdir(PROCESSED_FOLDER)):
        articles_ids, tags = get_info(topic, processed_news)
        # print(articles_ids)
        for id, tag in zip(articles_ids, tags):
            with open(f'{PROCESSED_FOLDER}/{topic}/{id}.txt', "w", encoding="utf-8") as file:
                file.write(tag)