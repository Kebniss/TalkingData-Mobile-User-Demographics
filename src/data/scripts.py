import numpy as np
import pandas as pd

def fillnan(arr):
    arr = pd.Series(arr)
    for i, elem in enumerate(arr):
        if elem == 'nan' or elem == 'NaN' or np.isnan(elem):
            arr[i] = -1
    return arr

def inverse_map(elem, map_array):
        return map_array[elem]

def flatten_list(l):
    #res = [item for sublist in l for item in sublist if isinstance(item, (int, long)) item = list(item)]
    res = []
    l = [sublist if isinstance(sublist, list) else [sublist]
         for sublist in l]
    res = [item for sublist in l for item in sublist]

    return res
