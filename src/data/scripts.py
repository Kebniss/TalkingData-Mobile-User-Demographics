import numpy as np
import pandas as pd

def fillnan(df, ignore_columns=None):
    df = pd.DataFrame(df)
    process_columns = [col for col in df.columns if col not in ignore_columns]
    for c in process_columns:
        for i, elem in enumerate(df[c]):
            if elem == 'nan' or elem == 'NaN' or np.isnan(float(elem)):
                df.loc[i, c] = -1
    return df

def inverse_map(elem, map_array):
        return map_array[elem]

def flatten_list(l):
    #res = [item for sublist in l for item in sublist if isinstance(item, (int, long)) item = list(item)]
    res = []
    l = [sublist if isinstance(sublist, list) else [sublist]
         for sublist in l]
    res = [item for sublist in l for item in sublist]

    return res
# fillnan now processes a dataframe and can be told to ignore columns
