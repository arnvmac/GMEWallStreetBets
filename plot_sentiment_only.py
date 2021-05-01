import pandas as pd
import pdb
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt

gme_prices_sentiment = pd.read_csv('Data/predicted_percent_sentiment_by_day.csv')
gme_prices_predicted_sentiment = [np.nan, np.nan]
for i, row in gme_prices_sentiment.iterrows():
	if i < 2:
		continue
	gme_prices_predicted_sentiment.append(row['Close'] * (row['predicted_percent_change'] + 1)) 
gme_prices_sentiment['predicted_new_price'] = gme_prices_predicted_sentiment
plt.plot('index', 'Close', data=gme_prices_sentiment, marker='', label="Actual Close Price")
plt.plot('index', 'predicted_new_price', data=gme_prices_sentiment, marker='', label="Predicted Close Price with just Sentiment")
plt.axvline(x=39, color='r', linestyle='-') ## put the location where validation set begins
plt.legend()
plt.show()

