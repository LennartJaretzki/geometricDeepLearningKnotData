import pandas as pd

# FEATURE = "Fibered"# "Q-Positive" # "Fibered"

knotinfo = pd.DataFrame(pd.read_csv("./datasets/knotinfo.csv", sep = ",", header = 0, index_col = False))


import json
from planarDiagram import Knot
import matplotlib.pyplot as plt
import knotGraph
import networkx as nx
import pandas as pd
from collections import Counter
import numpy as np


# path = "./datasets/fragments/augmented_knotinfo_5.csv"
# path = "./datasets/result.csv"
# path = "./datasetm(/meanAbove71WithGraph.csv"
path = "./datasets/unknots.csv"

# path = "./datasets/knotinfo.csv"

knot_table = pd.read_csv(path)#[0:1]#[2978*2-3:2978*2+3]
knot_table = knot_table.reset_index(drop=True)

knot_table["size"] = knot_table["PD Notation"].map(
    lambda knot:json.loads(knot.replace(";",","))
).apply(len)

knot_distances = []
mean = np.mean(knot_table["size"])
print("mean", mean, np.std(knot_table["size"]))