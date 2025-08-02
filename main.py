import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

#Using GPU for model if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading the stock data
ticker = 'ULVR'
df = yf.download(ticker, '2020-01-01')

#Preprcoessing the data to be in a normal distribution and we are only using the close price
scaler = StandardScaler()
df['Close'] = scaler.fit_transform(df['Close'])

#Looking at just the data for the past 30 days, taking 29 input values and 1 output value
seq_length = 30
data = []
for i in range(len(df) - seq_length):
    data.append(df['Close'][i:i + seq_length].values)
data = np.array(data)
train_size = int(len(data) * 0.8)

#splitting the data into training and testing sets
X_train = torch.from_numpy(data[:train_size, :-1, :])
y_train = torch.from_numpy(data[:train_size, -1, :])
X_test = torch.from_numpy(data[train_size:, :-1, :])      
y_test = torch.from_numpy(data[train_size:, -1, :])