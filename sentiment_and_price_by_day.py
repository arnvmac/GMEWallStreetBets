import pandas as pd
import pdb
import numpy as np

sentiments_by_day = pd.read_csv('Data/sentiment_by_day.csv')
get_percent_change = pd.read_csv('Data/percentage_change_by_day.csv')
print(sentiments_by_day.iloc[0,:])
print(get_percent_change.iloc[0,:])
merged_data = get_percent_change.merge(sentiments_by_day, on='date', how='inner')
merged_data.to_csv('Data/sentiment_price_each_day.csv', index=False)