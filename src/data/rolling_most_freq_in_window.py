import pandas as pd
from count_list_and_int import count_list_and_int
from collections import Counter

def rolling_most_freq_in_window(df,
                                groupby_key,
                                col_to_roll,
                                windows=[2, 3, 7, 10]
                                ):

    col_to_roll = 'active_apps_cat'
    df = daily_active_cat[['device_id','active_apps_cat']]
    groupby_key = 'device_id'
    windows=[2, 3, 7, 10]
    df[groupby_key], map_id = pd.factorize(df.device_id)
    if 'most_freq_app_dly' in df:
        df['most_freq_app_dly'], map_id = pd.factorize(df.most_freq_app_dly)

    df = df.reset_index(groupby_key).groupby(groupby_key, as_index=False)

    # aggr = pd.DataFrame()
    # tmp = pd.DataFrame()
    # windows = [2, 3, 7, 10]
    # for i in windows:
    #     for key, group in df:
    #         arr = [group.most_freq_app_dly.shift(x).values[::-1][:i] for x in range(len(group))[::-1]]
    #         most_recent = arr[-1]
    #         most_used = Counter(most_recent).most_common()[0][0]
    #         time = group['timestamp'].max()
    #         if i == 2:
    #             tmp = tmp.append([[key, time, most_used]], ignore_index=True)
    #         else:
    #             tmp = tmp.append([[most_used]], ignore_index=True)
    #     if i ==2:
    #         aggr = tmp
    #         aggr.columns = [groupby_key, 'timestamp', '2days_most_used_app']
    #     else:
    #         aggr[str(i) + 'days_most_used_app'] = tmp
    def inverse_map(elem, map_array):
            return map_array[elem]

    def flatten_list(l):
        #res = [item for sublist in l for item in sublist if isinstance(item, (int, long)) item = list(item)]
        res = []
        for sublist in l:
            if isinstance(sublist, (int, long)):
                sublist = list(sublist)
            for item in sublist:
                if not res:
                    res = [item]
                else:
                    res.append(item)
        return res

    key = 0
    group = df.get_group(key)
    aggr_2d = pd.DataFrame()
    for key, group in df:
        arr = [group[col_to_roll].shift(x).values[::-1][:2] for x in range(len(group))[::-1]]
        most_recent = arr[-1]
        most_recent = flatten_list(most_recent)
        most_used = Counter(most_recent).most_common()[0][0]
        time = group.timestamp.max()
        most_used = inverse_map(most_used, map_id)
        aggr_2d = aggr_2d.append([[key, time, most_used]])

    aggr_2d.columns=[groupby_key, 'timestamp', '2days_most_used']

    aggr_3d = pd.DataFrame()
    for key, group in df:
        arr = [group[col_to_roll].shift(x).values[::-1][:3] for x in range(len(group))[::-1]]
        most_recent = arr[-1]
        most_recent = flatten_list(most_recent)
        most_used = Counter(most_recent).most_common()[0][0]
        most_used = inverse_map(most_used, map_id)
        time = group.timestamp.max()
        aggr_3d = aggr_3d.append([[key, time, most_used]])

    aggr_3d.columns=[groupby_key, 'timestamp', '3days_most_used']

    aggr_7d = pd.DataFrame()
    for key, group in df:
        arr = [group[col_to_roll].shift(x).values[::-1][:7] for x in range(len(group))[::-1]]
        most_recent = arr[-1]
        most_recent = flatten_list(most_recent)
        most_used = Counter(most_recent).most_common()[0][0]
        most_used = inverse_map(most_used, map_id)
        time = group.timestamp.max()
        aggr_7d = aggr_7d.append([[key, time, most_used]])

    aggr_7d.columns=[groupby_key, 'timestamp', '7days_most_used']

    aggr_10d = pd.DataFrame()
    for key, group in df:
        arr = [group[col_to_roll].shift(x).values[::-1][:10] for x in range(len(group))[::-1]]
        most_recent = arr[-1]
        most_recent = flatten_list(most_recent)
        most_used = Counter(most_recent).most_common()[0][0]
        most_used = inverse_map(most_used, map_id)
        time = group.timestamp.max()
        aggr_10d = aggr_10d.append([[key, time, most_used]])



    aggr_10d.columns=[groupby_key, 'timestamp', '10days_most_used']

    rolled_df = aggr_2d.merge(aggr_3d.drop('timestamp',1),
                              on=groupby_key,
                              how='left')
    rolled_df = rolled_df.merge(aggr_7d.drop('timestamp',1),
                           on=groupby_key,
                           how='left')
    rolled_df = rolled_df.merge(aggr_10d.drop('timestamp',1),
                           on=groupby_key,
                           how='left')

    rolled_df[groupby_key] = rolled_df[groupby_key].apply(lambda x:
                                                          inverse_map(x,
                                                                      map_dev_id))


    return rolled_df
