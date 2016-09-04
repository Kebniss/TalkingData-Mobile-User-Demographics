def rolling_stats_in_window(df,
                            groupby_key,
                            aggs = ['mean', 'var'],
                            windows={'day':1, 'week':7, 'month':28, 'year':365},
                            ignore_columns = None):
    import pandas as pd
    
    df = df.sort_index()
    # Resample every 30 mins so that we can update after a game in the same
    # day
    grouped_df = df.groupby(groupby_key).resample('1D').mean()

    if not ignore_columns:
        ignore_columns = set([])
    else:
        ignore_columns = set(ignore_columns)
    ignore_columns.add(groupby_key)

    process_columns = [col for col in df.columns if col not in ignore_columns]

    aggs_dict = {process_col: aggs for process_col in process_columns}
    aggs_dict[groupby_key] = "count"

    rolled_df = pd.DataFrame()
    for window_name, window_value in windows.items():
        win_df = (grouped_df
                            .rolling(window_value, min_periods=1)
                            .agg(aggs_dict)
                            .shift(1) # at day i we know until i-1
                            )
        win_df.columns = ['_'.join(col).strip() + "_" + window_name
                          for col in win_df.columns.values]
        win_df = win_df.rename(
            columns = {"{0}_count_{1}".format(groupby_key, window_name) :
                       "count_{0}".format(window_name)
                       })
        if rolled_df.empty:
            rolled_df = win_df
        else:
            rolled_df = rolled_df.join(win_df, how='outer')
    return rolled_df
