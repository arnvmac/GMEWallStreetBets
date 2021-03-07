import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pdb
import pandas as pd
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
class SentimentByDay(Dataset):

    def __init__(self):
        xy = pd.read_csv('Data/sentiment_price_each_day.csv')
        self.n_samples = xy.shape[0]
        self.x_data = torch.from_numpy(xy[['daily_average_title_positive_sentiment',
        								'daily_average_title_negative_sentiment',
        								'daily_average_title_neutral_sentiment',
        								'daily_average_body_positive_sentiment',
        								'daily_average_body_negative_sentiment',
        								'daily_average_body_neutral_sentiment',
        								'volume_of_posts']].values)
        self.y_data = torch.from_numpy(xy[['gme_percentage_change']].values)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
# create dataset
dataset = SentimentByDay()
batch_size=4
train_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)



device = torch.device('cuda')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(7, 10, 1).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
writer = SummaryWriter('runs/gme_experiment_1')
count = 0
for epoch in range(100):
    for i, (features, labels) in enumerate(train_loader):  
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        labels = labels.to(device)
        features = features.to(device)
        # Forward pass
        outputs = model(features.float())
        loss = criterion(outputs, labels.float())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('training loss', loss, count )
        if (i+1) % 10 == 0 or 1 == 1:
            print (f'Epoch [{epoch+1}/{10}], Step [{i+1}], Loss: {loss.item()}')
        count += 1	
writer.close()
