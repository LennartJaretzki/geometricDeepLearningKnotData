import os
import pandas as pd

def concat():
    # folder_path = "./datasets/fragments"
    # folder_path = "./datasets/meanAbove71"
    # folder_path = "./datasets/unknots.csv"
    folder_path = "./datasets/unknotFragments/"


    dfs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            dfs.append(df)
    print(dfs,len(dfs))
    result_df = pd.concat(dfs, ignore_index=True)
    print(result_df)
    result_df.to_csv("unknots.csv", index=False)

import knotGraph
import networkx as nx
from planarDiagram import Knot
import json
def toGraphFile():
    for i in range(1,200):
        # path = f"./datasets/fragments/augmented_knotinfo_{i}.csv"
        path = f"./datasets/unknotFragments/unknots_{i}"
        # path = f"./datasets/knotinfo.csv"
        # path = f"./datasets/meanAbove71.csv"

        knot_table = pd.read_csv(path)#[2978*2-3:2978*2+3]
        knot_table = knot_table.loc[knot_table['PD Notation'].apply(lambda x: x != "[]")]
        print(knot_table)
        knot_table["graph"] = knot_table["PD Notation"].map(
            lambda knot_data: json.loads(knot_data.replace("(","[").replace(")","]"))
        ).apply(Knot).apply(knotGraph.graphRepresentation).apply(nx.node_link_data)
        knot_table["Crossing Number"] = 0 # minimal crossing number comment out when converting non unknot data

        # knot_table["graph"] = knot_table["PD Notation"].map(
        #     lambda knot:json.loads(knot.replace(";",","))
        # ).apply(Knot).apply(knotGraph.graphRepresentation).apply(nx.node_link_data)
        knot_table.to_csv(path, index=False)
        # print(i)

def concatUnknotWithKnot():
    unknot_df = pd.read_csv("./datasets/unknots.csv")[0:50000]
    ununknot_df = pd.read_csv("./datasets/resultWithGraph.csv")[["PD Notation", "graph", "Crossing Number"]][0:50000]
    print(unknot_df)
    print(ununknot_df)
    
    result_df = pd.concat([unknot_df, ununknot_df], ignore_index=True)
    result_df.to_csv("./datasets/unknotRecognition.csv", index=False)


if __name__ == '__main__':
    # toGraphFile()
    # concat()
    concatUnknotWithKnot()
