
from tqdm import tqdm
import os
import json
from src.crawler.utils import read_yaml
from src.crawler.getlinks import get_links_from_subtopics, get_content_from_article
 
def scrape_news():
    topics_links = read_yaml('./src/links.yaml')
    topics_links = get_links_from_subtopics(topics_links)
        
    # set output path
    OUTPUT = './data/raw_news'

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
    # scrape_news()
    
    import requests
    from bs4 import BeautifulSoup
    
    resp = requests.get('https://vnexpress.net/ong-khuat-viet-hung-dieu-khien-xe-may-dien-cung-can-bang-lai-4722702.html')
    # print(resp.text)
    soup = BeautifulSoup(resp.text, 'html.parser')
    comments = soup.find_all('label', id='total_comment')
    print(comments)
    