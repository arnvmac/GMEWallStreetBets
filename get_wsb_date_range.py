import pandas as pd
import pdb
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
import re
import numpy
import matplotlib.pyplot as plt
#commiting to GitHub
##nltk.download('vader_lexicon')
dataset = pd.read_csv('Data/archive_1/reddit_wsb.csv')

## gives a vector of true/false where true is rows which are NA
## select rows in dataset where body is na and set it equal to ''
dataset.loc[dataset['body'].isna(),'body'] = ''

dataset = dataset.dropna()

dataset.body = dataset.body.str.lower()

#Remove handlers
dataset.title = dataset.title.apply(lambda x:re.sub('@[^\s]+','',x))
dataset.body   = dataset.body.apply(lambda x:re.sub('@[^\s]+','',x))

# Remove URLS
dataset.title = dataset.title.apply(lambda x:re.sub(r"http\S+", "", x))
dataset.body   = dataset.body.apply(lambda x:re.sub(r"http\S+", "", x))

# Remove all the special characters
## It's -> It (possibly fix later)
dataset.title = dataset.title.apply(lambda x:' '.join(re.findall(r'\w+', x)))
dataset.body   = dataset.body.apply(lambda x:' '.join(re.findall(r'\w+', x)))

#remove all single characters
dataset.title = dataset.title.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
dataset.body   = dataset.body.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

# Substituting multiple spaces with single space
dataset.title = dataset.title.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
dataset.body   = dataset.body.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))


sid = SIA()
dataset['body_sentiments'] = dataset['body'].apply(lambda x: sid.polarity_scores(' '.join(re.findall(r'\w+',x.lower()))))
dataset['title_sentiments'] = dataset['title'].apply(lambda x: sid.polarity_scores(' '.join(re.findall(r'\w+',x.lower()))))
#r_data = r_data[pd.to_datetime(r_data.timestamp).dt.year>=2021]
#pre process code and remove handlers, urls, etc
dataset['positive_title_sentiment']   = dataset['title_sentiments'].apply(lambda x: x['pos']) 
dataset['neutral_title_sentiment']    = dataset['title_sentiments'].apply(lambda x: x['neu'])
dataset['negative_title_sentiment']   = dataset['title_sentiments'].apply(lambda x: x['neg'])

dataset['positive_body_sentiment']   = dataset['body_sentiments'].apply(lambda x: x['pos']) 
dataset['neutral_body_sentiment']    = dataset['body_sentiments'].apply(lambda x: x['neu'])
dataset['negative_body_sentiment']   = dataset['body_sentiments'].apply(lambda x: x['neg'])

plt.scatter('neutral_title_sentiment','positive_body_sentiment')
plt.show()