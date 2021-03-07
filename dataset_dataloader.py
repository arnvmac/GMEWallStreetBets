import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class SentimentByDay(Dataset):

    def __init__(self):
        xy = np.loadtxt('Data/sentiment_by_day.csv', delimiter=',', dtype=None, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x_data = torch.from_numpy(xy[:, 1:]) 
        self.y_data = torch.from_numpy(xy[:, [0]]) 

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# create dataset
SentimentByDay()

