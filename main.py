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
X_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)      
y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

#Defining the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__() 
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

#Passing the data to the model
model = LSTM(input_size=1, hidden_size=32, num_layers=2, output_size=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200
for i in range(num_epochs):
    y_training_pred = model(X_train)
    loss = criterion(y_training_pred, y_train)
    if i % 25 == 0:
        print(f'Epoch [{i}/{num_epochs}], Loss: {loss.item():.4f}')
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()