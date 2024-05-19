## Introduction
Implement the word embedding for exploring the correlation among words - Design a sequence model for generating text

* Work-flow

![image](https://github.com/Narius2030/Vietnamese-Text-Generator/assets/94912102/a7d6e12a-266f-42d9-8452-aade83844dbd)

## Implement
- Apply natural language processing techniques, such as: remove punctuations and symbols, remove stop words, reformat text, tokenize words and create corpus
- Design an artificial neural network model for generating text by using `LSTM`, I use Embedding layer for embedding word from text to feature vector and find relationships among them
- In the final layer, I use Dense layer with `Softmax` activation function for classifying which word has the highest probability
- Besides, I implement a data pipeline using Apache Airflow for crawling text data from VnExpress, I utilize the BeautifulSoup for `crawler`

### Buil Model
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 50, input_length=50))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LSTM(512, return_sequences=True))
model.add(tf.keras.layers.LSTM(512))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
```
```markdown
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_2 (Embedding)     (None, 50, 50)            190600    
                                                                 
 batch_normalization_4 (Bat  (None, 50, 50)            200       
 chNormalization)                                                
                                                                 
 lstm_4 (LSTM)               (None, 50, 512)           1153024   
                                                                 
 lstm_5 (LSTM)               (None, 512)               2099200   
                                                                 
 dense_9 (Dense)             (None, 100)               51300     
                                                                 
 dropout_2 (Dropout)         (None, 100)               0         
                                                                 
 batch_normalization_5 (Bat  (None, 100)               400       
 chNormalization)                                                
                                                                 
 dense_10 (Dense)            (None, 3812)              385012    
                                                                 
=================================================================
Total params: 3879736 (14.80 MB)
Trainable params: 3879436 (14.80 MB)
Non-trainable params: 300 (1.17 KB)
_______________________________________
```

### Data Scraping
```python
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
```

### Data pipeline in Airflow
```python
dag = DAG(
    'ETL-VNExpress',
    default_args={'start_date': days_ago(1)},
    schedule_interval='55 17 * * *',
    catchup=False
)

extract_data = PythonOperator(
    task_id='extract_data',
    python_callable=scrape_news,
    dag=dag
)

transform_load = PythonOperator(
    task_id='transform_load',
    python_callable=transform_load,
    dag=dag
)

print_date_task = PythonOperator(
    task_id='print_date',
    python_callable=print_date,
    dag=dag
)

# Set the dependencies between the tasks
extract_data >> transform_load >> print_date_task
```
