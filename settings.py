FEATURE = "Q-Positive"#"Fibered"# "" # "Fibered"
# FEATURE = "Ozsvath-Szabo tau"
# SPLIT_TYPE = "byCrossing"
# FEATURE = "Arf Invariant"
# FEATURE = "Fibered"
# FEATURE = "Determinant"
# FEATURE = "sign"
# FEATURE = "Double Slice Genus"
# FEATURE = "Volume"
# FEATURE = "Chern-Simons Invariant"
# FEATURE = "Rasmussen s"
# FEATURE = "Signature"
# FEATURE = "Genus-3D"
# FEATURE ="UNKNOT"	
ADDITIONAL = True

POOLING = "custom" # "custom"

# CONV_TYPE = "PNA"
# CONV_TYPE = "GAT"
CONV_TYPE = "TRANS"

LOSS = "abs"
# LOSS = "squared"

SPLIT_TYPE = "random"
# SPLIT_TYPE = "byCrossing"

PATH = "./datasets/knotinfoWithGraph.csv"
# PATH = "./datasets/meanAbove71WithGraph.csv"

# PATH = "./datasets/result.csv"
# PATH = "./datasets/resultWithGraph.csv"
BATCH_SIZE = 128#4*512#256#1024

GENERATE_GRPHS = False

# PATH = "./datasets/augmented_knotinfo_11.csv"

AUG_PATH = "./datasets/augmented_knotinfo.csv"


# SETTINGS FOR UNKNOT RECOGNITION
# FEATURE = "UNKNOT"
# ADDITIONAL = False
# GENERATE_GRPHS = False

# PATH = "./datasets/unknotRecognition.csv"

# assert not (FEATURE=="UNKNOT" and ADDITIONAL)
