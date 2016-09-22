import pandas as pd
""" Looks for nans in each column and drops those rows that have them"""

def drop_nans(df):

    cols = df.columns
    for _, col in enumerate(cols):
            nulls = df[df[col].isnull()]
            if not nulls.empty:
                df.drop(nulls, axis=1, inplace=True)

    return df
