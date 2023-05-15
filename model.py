# from math import tanh
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential, Sigmoid, Tanh
from torch_geometric.nn import (BatchNorm, PNAConv, global_add_pool,
                                global_max_pool, global_mean_pool)
from torch_geometric.nn.conv import GATConv, TransformerConv
import numpy as np
from settings import *

class NeuralKnotNet(torch.nn.Module):
    def __init__(self,deg):
        super().__init__()
        self.channels=20

        self.convs = ModuleList()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        # for _ in range(num_layers):
        #     nn = Sequential(
        #         Linear(channels, channels),
        #         ReLU(),
        #         Linear(channels, channels),
        #     )

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        if CONV_TYPE=="PNA":
            conv = PNAConv(in_channels=1, out_channels=self.channels,
                                aggregators=aggregators, scalers=scalers, deg=deg,
                                edge_dim=-1, towers=4, pre_layers=1, post_layers=1,
                                divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.channels))
            for _ in range(4):
                conv = PNAConv(in_channels=self.channels, out_channels=self.channels,
                                aggregators=aggregators, scalers=scalers, deg=deg,
                                edge_dim=-1, towers=4, pre_layers=1, post_layers=1,
                                divide_input=False)
                self.convs.append(conv)
                self.batch_norms.append(BatchNorm(self.channels))
            
            mlpInputSize = self.channels*3

        if CONV_TYPE=="GAT":
            self.heads = 1
            conv = GATConv(in_channels=1, out_channels=self.channels,
                                aggregators=aggregators, scalers=scalers, deg=deg,
                                edge_dim=-1, heads=self.heads,# pre_layers=1, post_layers=1,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.channels*self.heads))
            for _ in range(25):
                conv = GATConv(in_channels=self.channels*self.heads, out_channels=self.channels,
                                aggregators=aggregators, scalers=scalers, deg=deg,
                                edge_dim=-1, heads=self.heads,# pre_layers=1, post_layers=1,
                )
                for _ in range(1):
                    self.convs.append(conv)
                    self.batch_norms.append(BatchNorm(self.channels*self.heads))

            mlpInputSize = self.channels*3*self.heads

        if CONV_TYPE=="TRANS":
            self.heads = 3
            conv = TransformerConv(in_channels=1, out_channels=self.channels,
                                aggregators=aggregators, scalers=scalers, deg=deg,
                                edge_dim=-1, heads=self.heads,# pre_layers=1, post_layers=1,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.channels*self.heads))
            for _ in range(12):
                conv = TransformerConv(in_channels=self.channels*self.heads, out_channels=self.channels,
                                aggregators=aggregators, scalers=scalers, deg=deg,
                                edge_dim=-1, heads=self.heads,# pre_layers=1, post_layers=1,
                )
                for _ in range(1):
                    self.convs.append(conv)
                    self.batch_norms.append(BatchNorm(self.channels*self.heads))

            mlpInputSize = self.channels*3*self.heads

        
        if ADDITIONAL:
            mlpInputSize+=10
        self.mlp = Sequential(
            Linear(mlpInputSize, 25),
            BatchNorm(25), Tanh(), # 3 because of the maximum,average,summation channels
            Linear(25, 25), Tanh(),
            Linear(25, 25), Tanh(),
            Linear(25, 25), ReLU(),
            Linear(25, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch, additionalFeatures=None):
        # x = self.node_emb(x.squeeze(-1)) + self.pe_lin(pe)
        # edge_attr = self.edge_emb(edge_attr)

        # for conv in self.convs:
            # x = conv(x, edge_index, edge_attr=edge_attr)
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            # x = conv(x, edge_index, edge_attr)
            # x = batch_norm(
            #                 conv(x, edge_index, edge_attr)
            # )
            x = torch.tanh(
                    batch_norm(
                            conv(x, edge_index, edge_attr)
                    )
                )

        # x = global_mean_pool(x, batch)
        # print(len(x),len(x[0]),len(x.mean(dim=1)))
        # x=torch.flip(x,(1,0))
        # print(len(x),len(x[0]),len(x.mean(dim=1)))
        # print(len(x),len(x[0]),x.view(self.channels,-1).max(1))
        
        # maximum,_ = x.view(self.channels,-1).max(1)
        # minimum,_ = x.view(self.channels,-1).max(1)
        # average = x.view(self.channels,-1).mean(1)
        # x = torch.cat([maximum,minimum,average])
        
        # elif POOLING=="custom":
        #     maximum,_ = x.view(self.channels,-1).max(1)
        #     minimum,_ = x.view(self.channels,-1).max(1)
        #     average = x.view(self.channels,-1).mean(1)
        #     x = torch.cat([maximum,minimum,average])

        summation = global_add_pool(x, batch)
        average = global_mean_pool(x, batch)
        maximum = global_max_pool(x, batch)
        
        x = torch.cat([maximum,average,summation],dim=1)
        
        # x = torch.zeros(x.shape)
        if ADDITIONAL:
            x = torch.cat([x,additionalFeatures],dim=1)
        # print(type(x),type(additionalFeatures))
        # x=additionalFeatures
        x = self.mlp(x)
        return x