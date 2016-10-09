import pandas as pd

def rolling_stats_in_window(df,
                            groupby_key,
                            counting_key,
                            aggs = ['mean', 'var'],
                            windows={'30D':30, '365D': 365},
                            ignore_columns = None,
                            resample_value='1D'
                            ):
    df = df.sort_index()
    # Resample every 30 mins so that we can update after a game in the same
    # day
    if resample_value is not None:
        resampled_df = df.groupby(groupby_key).resample(resample_value).mean()
    else:
        resampled_df = df
    grouped_df = resampled_df.reset_index(groupby_key).groupby(groupby_key)

    if not ignore_columns:
        ignore_columns = set([])
    else:
        ignore_columns = set(ignore_columns)
    ignore_columns.add(groupby_key)

    process_columns = [col for col in df.columns if col not in ignore_columns]

    aggs_dict = {agg: process_columns for agg in aggs}
    aggs_dict['count'] = [counting_key]

    rolled_df = pd.DataFrame()
    for window_name, window_value in windows.items():
        win_df = pd.DataFrame()
        for agg, columns in aggs_dict.items():
            agg_df = grouped_df[columns].rolling(window_value, min_periods=1)
            if agg == 'mean':
                agg_df = agg_df.mean()
            elif agg == 'var':
                agg_df = agg_df.var()
            elif agg == 'std':
                agg_df = agg_df.std()
            elif agg == 'min':
                agg_df = agg_df.min()
            elif agg == 'max':
                agg_df = agg_df.max()
            elif agg == 'count':
                agg_df = agg_df.count()
            else:
                raise ValueError('Aggregate {0} unknown.'.format(agg) )

            agg_df = agg_df.shift(1) # at day i we know until i-1
            agg_df.columns = [col.strip()
                              + "_" + agg
                              + "_" + window_name
                              for col in agg_df.columns.values]
            if win_df.empty:
                win_df = agg_df
            else:
                win_df = win_df.join(agg_df, how='outer')

        win_df = win_df.rename(
            columns = {"{0}_count_{1}".format(counting_key, window_name) :
                       "count_{0}".format(window_name)
                       })
        if rolled_df.empty:
            rolled_df = win_df
        else:
            rolled_df = rolled_df.join(win_df, how='outer')

    return rolled_df
