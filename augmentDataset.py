from spherogram import Link
from spherogram.links.simplify import reidemeister_III, possible_type_III_moves
import random
import pandas as pd
from settings import *
import json


def randomType3ReidemeisterMove(link):
    possible = possible_type_III_moves(link)
    if possible==[]:
        #print("No random type 3 move possible")
        return
    reidemeister_III(link, random.choice(possible))

def augmentKnot(PD_code,complexity=45):

    knot = Link(PD_code)
    knot.backtrack(complexity*2)
    for _ in range(complexity):
        knot.backtrack(5,prob_type_1=0.2, prob_type_2=0.8)
        for _ in range(int(complexity/20)):
            randomType3ReidemeisterMove(knot)
    knot._rebuild(same_components_and_orientations=True)

    knot.simplify()
    print(len(PD_code),len(knot.PD_code()))
    if len(PD_code)==len(knot.PD_code()):# or len(knot.PD_code())<100:
        return augmentKnot(PD_code,complexity=complexity+10)
    return knot.PD_code()



def formatPD_Notation(PD_code):
    return json.dumps(PD_code).replace(",",";")

def augmentData(shuffledKnots,savePath):
    knotinfo = pd.DataFrame(pd.read_csv(PATH, sep = ",", header = 0, index_col = False))[2900:]
    knotinfo["knot"] = knotinfo["PD Notation"].map(lambda knot:json.loads(knot.replace(";",",")))

    
    augmented_dataframes = []
    for _ in range(shuffledKnots):
        knotinfo_augmented = knotinfo.copy()
        knotinfo_augmented["PD Notation"] = knotinfo_augmented["knot"].apply(augmentKnot).apply(formatPD_Notation)
        # knotinfo_augmented["knot"]
        augmented_dataframes.append(knotinfo_augmented)
    del knotinfo_augmented["knot"]
    
    # Concatenate all augmented dataframes together along with the original dataframe
    # augmented_dataframes.append(knotinfo)
    concatenated_df = pd.concat(augmented_dataframes, axis=0, ignore_index=True)
    # concatenated_df.to_csv("./datasets/augmented_knotinfo_34.csv", index=False)
    concatenated_df.to_csv("./datasets/"+savePath, index=False)


def generateUnknotData(generatedUnknots,savePath):
    

    
    unknots = []
    for _ in range(generatedUnknots):
        print(_)
        unknots.append([augmentKnot([[0,0,1,1]], complexity=65)])
    
    unknotDf = pd.DataFrame(unknots,columns=["PD Notation"])
    unknotDf.to_csv("./datasets/unknotFragments/"+savePath, index=False)





if __name__ == '__main__':
    # augmentData(79,"unknots_1.csv")
    generateUnknotData(6000, "unknots_31")