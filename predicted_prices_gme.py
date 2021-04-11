import pandas as pd
import pdb
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import math

gme_prices = pd.read_csv('Data/predicted_percent_sentiment_by_day.csv')
gme_prices_close = pd.read_csv('Data/predicted_percent_close_only_by_day.csv')
gme_prices_close = gme_prices_close.rename(columns={'predicted_percent_change': 'predicted_percent_change_close'})
assert(all(gme_prices['Close'] == gme_prices_close['Close']))
gme_prices_close_subset = gme_prices_close[['date', 'predicted_percent_change_close']]
merged_data = gme_prices.merge(gme_prices_close_subset, on='date', how='inner')
gme_prices_predicted = [np.nan, np.nan]
gme_prices_predicted_close = [np.nan, np.nan]
for i, row in merged_data.iterrows():
	if i < 2:
		continue
	gme_prices_predicted.append(row['Close'] * (row['predicted_percent_change'] + 1)) 
	gme_prices_predicted_close.append(row['Close'] * (row['predicted_percent_change_close'] + 1)) 
gme_prices['predicted_new_price'] = gme_prices_predicted
gme_prices['predicted_new_price_close'] = gme_prices_predicted_close
gme_prices['index'] = gme_prices.index
import matplotlib.pyplot as plt
plt.plot('index', 'Close', data=gme_prices, marker='')
plt.plot('index', 'predicted_new_price', data=gme_prices, marker='')
plt.plot('index', 'predicted_new_price_close', data=gme_prices, marker='')
plt.axvline(x=29, color='r', linestyle='-') ## put the location where validation set begins
plt.legend()
plt.show()