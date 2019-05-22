import pandas as pd
data = pd.read_csv('health_train_data.csv', error_bad_lines=False)
data_text = data[['tweet']]
data_text['index'] = data_text.index
documents = data_text

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import *
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


words = []


processed_docs = documents['tweet'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

k=[]
file1 = open("tweets_all.txt","r+")

for i in file1:
        bow_vector = dictionary.doc2bow(preprocess(i))
        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
                if score >0.20:
                        k.append(i)
print(list(set(k)))
file1.close()
f=open('filteredtweets.txt','a+')
for d in k:
        f.write(d)
f.close()

