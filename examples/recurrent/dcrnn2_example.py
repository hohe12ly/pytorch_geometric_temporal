import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN2
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import METRLADatasetLoader

# GPU support
DEVICE = torch.device('cuda') # cuda

rootdir = '/mnt/data/home/yxl/test/pytorch_geometric_temporal'
datadir = rootdir + '/data'
runname = 'dcrnn_la'
rundir = rootdir + '/runs/' + runname
if not os.path.exists(rundir):
    os.makedirs(rundir)
os.chdir(rundir)
print('Working directory: ' + os.getcwd())

# time series parameters: time window is 12, prediction window is 1
x_winsize = 12
y_winsize = 1

# data loader
loader = METRLADatasetLoader(raw_data_dir=datadir)

# time window is 12, predicting 1 step ahead
dataset = loader.get_dataset(num_timesteps_in = x_winsize, num_timesteps_out = y_winsize)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

print("Number of train buckets: ", len(list(train_dataset)))
print("Number of test buckets: ", len(list(test_dataset)))

# tensor data parameters
shuffle=True
batch_size = 64

train_input = np.array(train_dataset.features) # (27399, 207, 2, 12)
train_target = np.array(train_dataset.targets) # (27399, 207, 12)
train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle,drop_last=True)
print('train data:', 'x dim:', train_input.shape, 'label dim:', train_target.shape)

test_input = np.array(test_dataset.features) # (, 207, 2, 12)
test_target = np.array(test_dataset.targets) # (, 207, 12)
test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle,drop_last=True)
print('test data:', 'x dim:', test_input.shape, 'label dim:', test_target.shape)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, input_dim, rnn_units, batch_size, timewin_size, pred_dim, K):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN2(in_channels=input_dim, out_channels=rnn_units, batch_size=batch_size, timewin_size=timewin_size, K=K)
        self.linear = torch.nn.Linear(rnn_units, pred_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

# model parameters
input_dim = 2
rnn_units = 64
pred_dim = y_winsize
K = 1

print(RecurrentGCN(input_dim, rnn_units, batch_size, x_winsize, pred_dim, K))

# training parameters
lr = 0.01
epsilon = 1e-3

model = RecurrentGCN(input_dim, rnn_units, batch_size, x_winsize, pred_dim, K).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=epsilon)

loss_fn = torch.nn.MSELoss()

print('Net\'s state_dict:')
total_param = 0
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    total_param += np.prod(model.state_dict()[param_tensor].size())
print('Net\'s total params:', total_param)
#--------------------------------------------------
print('Optimizer\'s state_dict:')
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name])

for snapshot in train_dataset:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    static_edge_weight = snapshot.edge_attr.to(DEVICE)
    break;

# training parameters
num_epochs = 100

model.train()

for epoch in range(num_epochs):
    step = 0
    loss_list = []
    avgloss = 0
    for encoder_inputs, labels in train_loader:
        # x = encoder_inputs[0].to(DEVICE) # non-batch version: (N, F, T)
        # y = labels[0].to(DEVICE) # non-batch version: (N, T)
        x = encoder_inputs.to(DEVICE)
        y = labels.to(DEVICE)
        y_hat = model(x, static_edge_index, static_edge_weight)         # Get model predictions
        loss = loss_fn(y_hat, y) # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step= step+ 1
        avgloss = loss.item() / step + (step-1) / step * avgloss
        loss_list.append([epoch, step, loss.item(), avgloss])

        if step % 20 == 0 :
            print('epoch', epoch, 'step', step, 'avg_loss:', avgloss)
    print("Epoch {} train RMSE: {:.4f}".format(epoch, avgloss))

torch.save(model.state_dict(), 'model.pt')

df_train_loss = pd.DataFrame(loss_list, columns=['epoch', 'step', 'loss', 'avg_loss'])
df_train_loss.to_csv('train_loss.csv', index=False)
df_train_loss['avg_loss'].plot().get_figure().savefig('train_loss.png')

# evaluation
model.eval()
step = 0
# Store for analysis
total_loss = []
test_labels = []
predictions = []
for encoder_inputs, labels in test_loader:
    # Get model predictions
    # x = encoder_inputs[0].to(DEVICE) # non-batch version: (N, F, T)
    # y = labels[0].to(DEVICE) # non-batch version: (N, T)
    x = encoder_inputs.to(DEVICE)
    y = labels.to(DEVICE)
    y_hat = model(x, static_edge_index, static_edge_weight)
    # Mean squared error
    loss = loss_fn(y_hat, y)
    total_loss.append(loss.item())
    # Store for analysis below
    test_labels.append(y.cpu().numpy())
    predictions.append(y_hat.detach().cpu().numpy())
    

print("Test MSE: {:.4f}".format(sum(total_loss)/len(total_loss)))

pd.DataFrame({'loss':total_loss}).to_csv('test_loss.csv', index=False)

plt.plot(total_loss)
plt.savefig('test_loss.png')

sensors = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
for sensor in sensors:
    timestep = 0 
    preds = np.asarray([pred[sensor][timestep].detach().cpu().numpy() for pred in y_hat])
    labs  = np.asarray([label[sensor][timestep].cpu().numpy() for label in labels])
    #print("Data points:,", preds.shape)

    plt.figure(figsize=(20,5))
    sns.lineplot(data=preds, label="pred")
    sns.lineplot(data=labs, label="true")
    plt.title("Sensor {} at next timestep {}".format(sensor, timestep + 1))
    plt.savefig('sensor_'+str(sensor)+'_pred.png')