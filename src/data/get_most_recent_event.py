import pandas as pd
import numpy as np

def get_most_recent_event(df,
                          groupby_key,
                          other_indexes):

    df = df.groupby(groupby_key)

    # Get the most recent statistic
    result = pd.DataFrame()
    for key, group in df:
        if result.empty:
            result = pd.DataFrame(group.ix[-1]).transpose()
        else:
            result = result.append( group.ix[-1].transpose() )

    result.index = result.index.rename(other_indexes)
    result[groupby_key] = result[groupby_key].astype(np.int64)
    result = result.reset_index().set_index([groupby_key, other_indexes])

    return result
