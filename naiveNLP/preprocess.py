import re 
from nltk.corpus import stopwords 
import nltk
import csv

nltk.download('stopwords')
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)?(?:-)?(?:\)|\(|D|P)', text)
    
    text = re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop] 
    
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            text = row[0]
            label = int(row[1])
            yield text, label

def get_minibatch_size(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    
    except StopIteration:
        
        return None, None 
    
    return docs, y