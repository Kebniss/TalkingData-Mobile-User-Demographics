import pandas as pd
import numpy as np

def get_most_recent_event(df,
                          groupby_key,
                          timestamp='timestamp'
                          ):

    if groupby_key in df.index.names:
        df = df.reset_index(groupby_key)
        df[groupby_key], map_id = pd.factorize(df[groupby_key])
        df = df.groupby(groupby_key)
    else:
        df = df.groupby(groupby_key)

    # Get the most recent statistic
    result = pd.DataFrame()
    for key, group in df:
        max_idx = group.loc[group.index.max()]
        if result.empty:
            result = pd.DataFrame(max_idx).T
        else:
            result = result.append( pd.DataFrame(max_idx).T )

    result = result.reset_index()
    result = result.rename(columns={ 'index': timestamp})
    result[groupby_key] = result[groupby_key].apply(lambda x:map_id[x])
    result = result.set_index([groupby_key, timestamp])

    return result
