import pandas as pd
from operations_on_list import count_list_and_int
from collections import Counter

def rolling_most_freq_in_window(df,
                                groupby_key,
                                col_to_roll,
                                windows=[2, 3, 7, 10]
                                ):

    df = df.reset_index()
    df[groupby_key], map_id = pd.factorize(df[groupby_key])
    map_id_apps = None
    if 'most_freq_app_dly' in df:
        df['most_freq_app_dly'], map_id_apps = pd.factorize(df['most_freq_app_dly'])

    df = df.groupby(groupby_key, as_index=False)

    def inverse_map(elem, map_array):
            return map_array[elem]

    def flatten_list(l):
        #res = [item for sublist in l for item in sublist if isinstance(item, (int, long)) item = list(item)]
        res = []
        l = [sublist if isinstance(sublist, list) else [sublist]
             for sublist in l]
        res = [item for sublist in l for item in sublist]

        return res

    aggr_2d = pd.DataFrame()
    for key, group in df:
        arr = ( [ group[col_to_roll].shift(x).values[::-1][:2]
                  for x in range(len(group))[::-1] ] )
        most_recent = arr[-1]
        most_recent = flatten_list(most_recent)
        most_used = Counter(most_recent).most_common()[0][0]
        time = group.timestamp.max()
        aggr_2d = aggr_2d.append([[key, time, most_used]])

    aggr_2d.columns=[groupby_key, 'timestamp', '2days_most_used']

    aggr_3d = pd.DataFrame()
    for key, group in df:
        arr = [group[col_to_roll].shift(x).values[::-1][:3] for x in range(len(group))[::-1]]
        most_recent = arr[-1]
        most_recent = flatten_list(most_recent)
        most_used = Counter(most_recent).most_common()[0][0]
        time = group.timestamp.max()
        aggr_3d = aggr_3d.append([[key, time, most_used]])

    aggr_3d.columns=[groupby_key, 'timestamp', '3days_most_used']

    aggr_7d = pd.DataFrame()
    for key, group in df:
        arr = [group[col_to_roll].shift(x).values[::-1][:7] for x in range(len(group))[::-1]]
        most_recent = arr[-1]
        most_recent = flatten_list(most_recent)
        most_used = Counter(most_recent).most_common()[0][0]
        time = group.timestamp.max()
        aggr_7d = aggr_7d.append([[key, time, most_used]])

    aggr_7d.columns=[groupby_key, 'timestamp', '7days_most_used']

    aggr_10d = pd.DataFrame()
    for key, group in df:
        arr = [group[col_to_roll].shift(x).values[::-1][:10] for x in range(len(group))[::-1]]
        most_recent = arr[-1]
        most_recent = flatten_list(most_recent)
        most_used = Counter(most_recent).most_common()[0][0]
        time = group.timestamp.max()
        aggr_10d = aggr_10d.append([[key, time, most_used]])

    aggr_10d.columns=[groupby_key, 'timestamp', '10days_most_used']

    tmp = rolled_df

    rolled_df = aggr_2d.merge(aggr_3d.drop('timestamp',1),
                              on=groupby_key,
                              how='left')
    rolled_df = rolled_df.merge(aggr_7d.drop('timestamp',1),
                           on=groupby_key,
                           how='left')
    rolled_df = rolled_df.merge(aggr_10d.drop('timestamp',1),
                           on=groupby_key,
                           how='left')

    if map_id_apps is not None:
        rolled_df['2days_most_used'] = inverse_map(
                                                rolled_df['2days_most_used'],
                                                map_id_apps)
        rolled_df['3days_most_used'] = inverse_map(
                                                rolled_df['3days_most_used'],
                                                map_id_apps)
        rolled_df['7days_most_used'] = inverse_map(
                                                rolled_df['7days_most_used'],
                                                map_id_apps)
        rolled_df['10days_most_used'] = inverse_map(
                                                rolled_df['10days_most_used'],
                                                map_id_apps)

    rolled_df[groupby_key] = rolled_df[groupby_key].apply(lambda x:
                                                          inverse_map(x,
                                                                      map_id))
    return rolled_df
