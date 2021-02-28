import pandas as pd
import pdb
import numpy as np

gme_stock = pd.read_csv('Data/archive_2/GME_stock.csv')
gme_percentage_change = []
for i in range(gme_stock.shape[0]-1):
	close_price_today = gme_stock.loc[i, 'close_price']
	close_price_tmrw = gme_stock.loc[i+1, 'close_price']
	percentage_change = (close_price_tmrw-close_price_today)/close_price_today
	gme_percentage_change.append(percentage_change)
gme_percentage_change.append(np.nan)
gme_stock['gme_percentage_change'] = gme_percentage_change
gme_stock.to_csv('Data/percentage_change_by_day.csv', index = False)