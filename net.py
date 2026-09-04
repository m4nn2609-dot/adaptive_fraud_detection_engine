import torch.nn as nn
import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss,Sigmoid
from model import load_data


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_embedding=nn.Embedding(5,3)
        self.country_embedding=nn.Embedding(243,16)
        self.layer_1=nn.Linear(in_features=19+10,out_features=64)
        self.layer_2=nn.BatchNorm1d(num_features=64)
        self.relu1=nn.ReLU()
        self.dropout_1=nn.Dropout(p=0.2)
        self.layer_3=nn.Linear(in_features=64,out_features=20)
        self.layer_4=nn.BatchNorm1d(num_features=20)
        self.relu2=nn.ReLU()
        self.dropout_2=nn.Dropout(p=0.2)
        self.final=nn.Linear(in_features=20,out_features=1)
    def forward(self,x,device_idx,country_idx):
        temp_1=self.device_embedding(device_idx)
        temp_2=self.country_embedding(country_idx)
        x=torch.cat([x,temp_1,temp_2],dim=1)
        x=self.layer_1(x)
        x=self.layer_2(x)
        x=self.relu1(x)
        x=self.dropout_1(x)
        x=self.layer_3(x)
        x=self.layer_4(x)
        x=self.relu2(x)
        x=self.dropout_2(x)
        x=self.final(x)
        return x
def eval_params(x_train,y_train,x_test,y_test):
    negs = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    pos_weight = torch.tensor(negs / pos)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    train_df,test_df=load_data(x_train,y_train,x_test,y_test)
    model=net()
    return model,criterion,train_df,test_df

def train_net(y_train,train_df,net,criterion,epochs=10):
    optimiser = Adam(net.parameters(), lr=0.01)
    net.train()
    for epoch in range(epochs):
        for num_f, cat_f, label in train_df:
            optimiser.zero_grad()
            d_idx = cat_f[:, 0]
            c_idx = cat_f[:, 1]
            output = net(num_f, d_idx, c_idx)
            loss = criterion(output, label.view(-1, 1))
            loss.backward()
            optimiser.step()
        print(loss)
    return net
def eval_net(test_df,net,criterion):
    res = []
    net.eval()
    with torch.no_grad():
        for num_f, cat_f, label in test_df:
            d_idx = cat_f[:, 0]
            c_idx = cat_f[:, 1]
            output = net(num_f, d_idx, c_idx)
            loss = criterion(output, label.view(-1, 1))
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).int()
            res.append((label, pred))
    return model
