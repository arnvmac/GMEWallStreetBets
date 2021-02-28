import pandas as pd
import pdb
import numpy as np

wsb_sentiments = pd.read_csv('Data/sentiments_reddit_posts.csv')
features_dataframe = {'date': [], 'daily_average_title_positive_sentiment': [], 'daily_average_title_negative_sentiment': [],
'daily_average_title_neutral_sentiment': [], 'daily_average_body_positive_sentiment': [],
'daily_average_body_negative_sentiment': [], 'daily_average_body_neutral_sentiment': [], 'volume_of_posts': []}
wsb_sentiments['date'] = pd.to_datetime(wsb_sentiments['timestamp']).dt.date
for date in wsb_sentiments['date'].unique():
	# exatract all the rows from the wsb_sentiments for this 'date'
	wsb_sentiments_date = wsb_sentiments.loc[wsb_sentiments['date'] == date,:]
	daily_average_title_positive_sentiment = wsb_sentiments_date.loc[:, 'positive_title_sentiment'].mean()
	daily_average_title_negative_sentiment = wsb_sentiments_date.loc[:, 'negative_title_sentiment'].mean()
	daily_average_title_neutral_sentiment = wsb_sentiments_date.loc[:, 'neutral_title_sentiment'].mean()


	daily_average_body_positive_sentiment = wsb_sentiments_date.loc[:, 'positive_body_sentiment'].mean()
	daily_average_body_negative_sentiment = wsb_sentiments_date.loc[:, 'negative_body_sentiment'].mean()
	daily_average_body_neutral_sentiment = wsb_sentiments_date.loc[:, 'neutral_body_sentiment'].mean()


	volume_of_posts = wsb_sentiments_date.shape[0]

	features_dataframe['daily_average_title_positive_sentiment'].append(daily_average_title_positive_sentiment)
	features_dataframe['daily_average_title_negative_sentiment'].append(daily_average_title_negative_sentiment)
	features_dataframe['daily_average_title_neutral_sentiment'].append(daily_average_title_neutral_sentiment)

	features_dataframe['daily_average_body_positive_sentiment'].append(daily_average_body_positive_sentiment)
	features_dataframe['daily_average_body_negative_sentiment'].append(daily_average_body_negative_sentiment)
	features_dataframe['daily_average_body_neutral_sentiment'].append(daily_average_body_neutral_sentiment)

	features_dataframe['volume_of_posts'].append(volume_of_posts)
	features_dataframe['date'].append(date)
features_dataframe = pd.DataFrame(features_dataframe)
features_dataframe.to_csv('Data/sentiment_by_day.csv', index = False)
