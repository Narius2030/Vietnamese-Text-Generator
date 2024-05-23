
from tqdm import tqdm
import os
import json
from src.crawler.utils import read_yaml
from src.crawler.getlinks import get_links_from_subtopics, get_content_from_article
from function.preprocessing import NormalizeTexts
 
def scrape_news():
    topics_links = read_yaml('./src/crawler/links.yaml')
    topics_links = get_links_from_subtopics(topics_links, pages=3)
        
    # set output path
    OUTPUT = './data/vnexpress/raw_news'

    print('\nCrawling...')
    for topic, links in topics_links.items():
        # the number of news links per topic
        print(f'Topic {topic} - Number of Sub-topic: {len(links)}')
        
        # save news data into text file in raw_news folder
        file_path = os.path.join(OUTPUT, f'{topic}.txt')
        with open(file_path, 'w') as file:
            for link in tqdm(links):
                url = list(link.keys())[0]
                items = link[url]
                content = get_content_from_article(url, items[0], items[1], topic)
                if content is not None:
                    file.write(json.dumps(content))
                    file.write('\n')

    print('\nCompleted!')


if __name__ == '__main__':
    # scrape news from vnepxress
    # scrape_news()
    
    # clean text and save
    normalizer = NormalizeTexts()
    data = normalizer.load_data(path='./data/vnexpress/raw_news')
    cleaned_data = normalizer.transform_texts(data)
    normalizer.dump_files(cleaned_data, path='./data/vnexpress')