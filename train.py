
import matplotlib.pyplot as plt
# conv.TransformerConv
import numpy as np
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau

from settings import *
from model import NeuralKnotNet

import dataset
from torch_geometric.loader import DataLoader
train_loader, test_loader, deg = dataset.load()


testGeneralisazion = DataLoader(dataset.fetchDataset("datasets/meanAbove71WithGraph.csv")["dataPoint"], batch_size=BATCH_SIZE)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralKnotNet(deg=deg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)
print("Cuda is ", torch.cuda.is_available())
# print("Trainable parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))

def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        # print("mmh",data,len(data))
        # print(data.x,len(data.x))
        # print(data.batch)
        # print("y",data.y, data.featureEncoding)
        data = data.to(device)
        optimizer.zero_grad()
        if ADDITIONAL:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.featureEncoding)
        else:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        # print("out",out)
        # print(out.squeeze(), data.y)
        if LOSS=="abs":
            loss = (out.squeeze() - data.y).abs().mean()
        elif LOSS=="squared":
            loss = (out.squeeze() - data.y).square().mean()

        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    total_accuracy = torch.tensor(.0)
    for data in loader:
        data = data.to(device)
        if ADDITIONAL:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.featureEncoding)
        else:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_accuracy += (out.squeeze().round()==data.y).sum()#/len(out.squeeze())
        if LOSS=="abs":
            total_error += (out.squeeze() - data.y).abs().sum().item()
        elif LOSS=="squared":
            total_error += (out.squeeze() - data.y).square().sum().item()
    return total_error / len(loader.dataset), total_accuracy/ len(loader.dataset)


history_loss = []
history_val = []
history_acc=[]
for epoch in range(1, 500):
    val_loss, accuracy = test(test_loader)
    val_loss_generalization, accuracy_generalization = test(testGeneralisazion)
    loss = train(epoch)
    # test_mae = test(test_loader)

    history_loss.append(loss)
    history_val.append(val_loss)
    history_acc.append(accuracy)

    scheduler.step(loss)
    # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_loss:.4f}, Acc: {accuracy:.4f}')
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_loss:.4f}, Acc: {accuracy:.4f}, Val on big: {val_loss_generalization:.4f}, Acc on big: {accuracy_generalization:.4f}')


plt.plot(history_acc)
plt.plot(history_loss)
plt.plot(history_val)
plt.show()
