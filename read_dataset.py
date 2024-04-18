import numpy as np
path_collection = ''
data_folder = path_collection
data = np.load(path_collection, allow_pickle=True)

for dt in data:
    print(dt.keys())
    print(dt.heads())