import pandas as pd
import pdb
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cpu')
gme_prices_sentiments = pd.read_csv('Data/sentiment_price_each_day.csv')
gme_input_features = gme_prices_sentiments[['daily_average_title_positive_sentiment', 
						'daily_average_title_negative_sentiment', 'daily_average_title_neutral_sentiment', 
						'daily_average_body_positive_sentiment', 'daily_average_body_negative_sentiment',
						'daily_average_body_neutral_sentiment', 'volume_of_posts']]
## subset to features we want to use to predict the outcome and the outcome column

## for loop where we loop through each day we want to predict
## .. and for each day we want to predict from the previous x days gather the relevant input features
sequence_days = 3

class LSTM(nn.Module):
    ## input_dim: number of expected features in the input
    ## output_size: expected elements in the output
    ## hidden_dim: size of the 'hidden' state
    ## n_layers: layers in the LSTM if more than 1 this is a stacked LSTM
    def __init__(self, input_dim, output_size, hidden_dim, n_layers=1, drop_prob=0.5):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        self.hidden_dim = hidden_dim        
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
       
    def forward(self, x, hidden):
        batch_size = x.size(0)
        lstm_out, hidden = self.lstm(x, hidden)
        ## the output is batch x (sequence, hidden)
        ## .. here we just collapse it all into one many tables irrespective of batch
        ## .. the reason we do this is to feed it all through the feed forward network
        ## .. once we go through the feed forward network all the hidden_dim elements will collapse into just one output
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        #out = self.sigmoid(out)
        ## recreate the batch separation, we now have batch as row and sequence as columns, each element is a final prediction
        ## .. for each day in the sequence
        out = out.view(batch_size, -1)
        ## for each batch just select the final sequence output
        out = out[:,-1]
        return(out, hidden)
   
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return(hidden)


input_sequences = [] # information used to predict the label
output_labels = [] # label we want to predict
for i, row in gme_prices_sentiments.iterrows():
	# cant use the first x days beacuse we odnt have information
	if i < sequence_days-1:
		continue # skip to next iteration
	output_labels.append(gme_prices_sentiments[['gme_percentage_change']].iloc[i].values[0])
	# gather the relevant rows for the previous days
	sentiments_subset = gme_input_features.iloc[i-(sequence_days-1):i+1,:]
	input_sequences.append(sentiments_subset.values)

input_sequences_train = input_sequences
output_labels_train = output_labels

batch_size = 2
train_data = TensorDataset(torch.Tensor(input_sequences_train), torch.Tensor(output_labels_train))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

lstm = LSTM(input_dim = 7, output_size = 1, hidden_dim = 10)
loss_function = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
optimizer = optim.Adam(lstm.parameters(), lr=0.0001)
writer = SummaryWriter('runs/gme_experiment_4')
iteration = 0
for epoch in range(200):
	hidden = lstm.init_hidden(batch_size)
	for i, data in enumerate(train_loader, 0):
		iteration += 1
		inputs, labels = data
		optimizer.zero_grad()
		# recreates hidden variable so backpropogation gradient descent  is not influenced
		# by the previous hidden variable 
		hidden = tuple([e.data for e in hidden])
		outputs, hidden = lstm(inputs, hidden)
		loss = loss_function(outputs, labels)
		loss.backward()
		optimizer.step()
		writer.add_scalar('training loss', loss, iteration)
writer.close()