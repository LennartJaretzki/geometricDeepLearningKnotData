from knotGraph import graphRepresentation
from planarDiagram import Knot
import pandas as pd
import json
from settings import *
from torch_geometric.utils import degree, from_networkx
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import random
from copy import deepcopy as copy
import networkx as nx

def load():
    knot_dataset = fetchDataset()


    if SPLIT_TYPE=="byCrossing":
        train_dataset = knot_dataset[knot_dataset['Crossing Number'].apply(lambda crossingNumber: crossingNumber<12)]
        test_dataset = knot_dataset[knot_dataset['Crossing Number'].apply(lambda crossingNumber: crossingNumber>=12)]

    elif SPLIT_TYPE=="random":
        train_dataset, test_dataset=train_test_split(knot_dataset,test_size=0.2,train_size=0.8)
    # train_dataset = train_dataset[0:1600]


    # test_loader = DataLoader(knot_dataset, batch_size=16)
    # print(train_dataset)
    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    train_loader = DataLoader(train_dataset["dataPoint"], batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset["dataPoint"], batch_size=BATCH_SIZE)

    print("Dataset consists of ", len(train_dataset),"train samples and ", len(test_dataset), "test samples")

    # Compute the maximum in-degree in the training data.
    deg = getDegree(knot_dataset)
    
    return train_loader, test_loader, deg

def getDegree(knot_dataset):
    max_degree = -1
    for data in knot_dataset["dataPoint"]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    print("max_degree:",max_degree)
    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in knot_dataset["dataPoint"]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg

def fetchDataset(filePath = PATH):
    knotinfo = pd.DataFrame(pd.read_csv(filePath, sep = ",", header = 0, index_col = False))#[0:80000]
    
    knotinfo.reindex(np.random.permutation(knotinfo.index))
    knotinfo = knotinfo[0:9000]
    knot_dataset = knotinfo.reset_index(drop=True)

    if GENERATE_GRPHS:
        knotinfo["knot"] = knotinfo["PD Notation"].map(lambda knot:json.loads(knot.replace(";",","))).map(Knot)
    # knotinfo = knotinfo[0:200]
    

    if FEATURE not in ["sign", "UNKNOT"]:
        knotinfo["Determinant"]=(knotinfo["Determinant"]-knotinfo["Determinant"].min())/(knotinfo["Determinant"].max()-knotinfo["Determinant"].min())

    if FEATURE=="Chern-Simons Invariant":
        knotinfo = knotinfo[~knotinfo['Chern-Simons Invariant'].str.contains('Not Hyperbolic')]
        knotinfo.reindex()
    knot_dataset = pd.DataFrame()
    knot_dataset["Crossing Number"] = knotinfo["Crossing Number"]

    if FEATURE =="UNKNOT":
        knot_dataset["y"]=knotinfo["Crossing Number"].map(lambda number: 0 if number>0 else 1)
    elif FEATURE=="Q-Positive":
        knot_dataset["y"]=knotinfo["Q-Positive"].map(lambda feature: 0 if feature=="Y" else 1)
    elif FEATURE=="Fibered":
        knot_dataset["y"]=knotinfo["Fibered"].map(lambda feature: 0 if feature=="Y" else 1)
    elif  FEATURE=="Ozsvath-Szabo tau":
        knot_dataset["y"]=knotinfo["Ozsvath-Szabo tau"]#.map(lambda feature: 0 if feature<1 else 1)
    elif FEATURE == "Arf Invariant":
        knot_dataset["y"]=knotinfo["Arf Invariant"]
    elif FEATURE == "Double Slice Genus":
        def sliceToNum(genus):
            try:
                return int(genus)
            except ValueError:
                genus = json.loads(genus.replace(";",","))
                return int(genus[0])/2+int(genus[1])/2
        knot_dataset["y"]=knotinfo["Double Slice Genus"].map(lambda feature: 1 if sliceToNum(feature)==6 else 0)
    elif FEATURE == "determinant":
        knot_dataset["y"]=knotinfo["Determinant"]
    elif FEATURE=="Chern-Simons Invariant":
        knot_dataset["y"] = knotinfo["Chern-Simons Invariant"].apply(float)
    else:
        knot_dataset["y"]=knotinfo[FEATURE]

    if GENERATE_GRPHS:
        knot_dataset["graph"]=knotinfo["knot"].map(graphRepresentation)
    else:
        def makeReadable(json_dump): # neccessary because when pandas saved json object it saved it in a format that the standard python library cant read which lead to this bodge
            return json_dump.replace("'",'"').replace("False","false").replace("True","true").replace("(","[").replace(")","]")
        knotinfo["graph"] = knotinfo["graph"].apply(makeReadable)
        knotinfo["graph"] = knotinfo["graph"].apply(json.loads)
        # print(knotinfo["graph"][0], json.loads(knotinfo["graph"][0]))
        knot_dataset["graph"]=knotinfo["graph"].apply(lambda input_string: nx.node_link_graph(input_string))

    # alternating, fibered, a positive braid closure, or large or small,
    # together with integer-valued variables which encode the crossing number, Seifert
    # genus, braid index, signature, arc index, determinant, and Rasmussen invariant
    if ADDITIONAL:
        encodeBool = lambda feature: 0 if feature=="Y" else 1
        knot_dataset['featureEncoding'] = knotinfo.apply(lambda features: [
            encodeBool(features['Alternating']),
            encodeBool(features['Fibered']),
            encodeBool(features['Q-Positive Braid']),
            encodeBool(features['Small or Large']),
            features['Crossing Number'],
            features['Braid Index'],
            features['Signature'],
            features['Arc Index'],
            features['Determinant'],
            features['Rasmussen s']
        ], axis=1)

    del knotinfo
    knot_dataset["graph"]=knot_dataset["graph"].map(lambda graph: from_networkx(graph))
    # =group_node_attrs=nx.get_node_attributes(graph,"deltaSum")

    def convert_to_dataPoint(graph,y,featureEncoding=None):
        graph.y=y
        if ADDITIONAL:
            graph.featureEncoding = torch.asarray([
                featureEncoding
            ])
        return graph
    if ADDITIONAL:
        knot_dataset["dataPoint"]=[convert_to_dataPoint(*a) for a in tuple(zip(knot_dataset["graph"], knot_dataset["y"], knot_dataset["featureEncoding"]))]
    else:
        knot_dataset["dataPoint"]=[convert_to_dataPoint(*a) for a in tuple(zip(knot_dataset["graph"], knot_dataset["y"]))]

    knot_dataset.reindex(np.random.permutation(knot_dataset.index))
    knot_dataset = knot_dataset.reset_index(drop=True)
    return knot_dataset





if __name__ == '__main__':
    pass#augmentData()
    