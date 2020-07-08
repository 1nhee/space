import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame as df

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from prev_code import WeightedRMSE as WRMSE

print('data loading...')

# column_names = ['year', 'doy', 'hr', 'min', 'Np', 'Vp', 'Tp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt', 'Kp']
data = pd.read_csv('../organized_data/new_data_3hour_stats.csv',  delimiter=',')


# for val in range(0,10):
#     data[data['Kp'] == val] = data[data['Kp'] == val].fillna(data[data['Kp'] == val].mean(axis=0, skipna=True))

data_tr = data[data['year'] <= 2012]    ## originally 2010
data_ts = data[data['year'] > 2012]     ## originally 2010

Y_tr = data_tr.loc[:, data_tr.columns == 'Kp']
Y_ts = data_ts.loc[:, data_ts.columns == 'Kp']

X_tr = data_tr.loc[:, data_tr.columns != 'Kp']
X_ts = data_ts.loc[:, data_ts.columns != 'Kp']

X_tr = X_tr[X_tr.columns[4:]]
X_ts = X_ts[X_ts.columns[4:]]

# normalize
normalizer = StandardScaler()
normalizer.fit(X_tr)
print('normalizing...')
X_tr_normalized = normalizer.transform(X_tr)
X_ts_normalized = normalizer.transform(X_ts)


X_tr_normalized = torch.from_numpy(X_tr_normalized)
X_ts_normalized = torch.from_numpy(X_ts_normalized)

Y_tr = torch.FloatTensor(Y_tr['Kp'].values)
# Y_ts = torch.FloatTensor(Y_ts['Kp'].values)


X_tr_normalized = Variable(X_tr_normalized)
X_ts_normalized = Variable(X_ts_normalized)

Y_tr = Variable(Y_tr)
# Y_ts = Variable(Y_ts)


# this is one way to define a network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


net = Net(n_feature=35, n_hidden=20, n_output=1)     # define the network
# print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# train the network

for t in range(200):
    prediction = net(X_tr_normalized.float())  # input x and predict based on x

    loss = loss_func(prediction, Y_tr)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients


Y_ts_pred = net(X_ts_normalized.float())

print('RESULT(wrmse):')
WRMSE.wrmse(Y_ts_pred.detach().numpy(),Y_ts)

print('RESULT(wrmse)-rounded:')
WRMSE.wrmse(np.clip(np.round(Y_ts_pred.detach().numpy()), a_min=0, a_max=9),Y_ts)