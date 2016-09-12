def rolling_stats_in_window(df,
                            groupby_key,
                            aggs = ['mean', 'var', 'max'],
                            windows={'day':1, 'week':7, 'month':28, 'year':365},
                            ignore_columns = None):
    import pandas as pd

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
        # MEAN
        win_df_mean = (df.reset_index('device_id').groupby('device_id')
                        .rolling(window_value, min_periods=1)
                        .agg({'daily_distance':'mean'})
                        )
        win_df_mean.columns = [win_df_mean.columns.values + "_" + window_name + '_mean']
        win_df_mean = win_df_mean.rename(
            columns = {"{0}_count_{1}".format(groupby_key, window_name) :
                       "count_{0}".format(window_name)
                       })

        # VAR
        win_df_var = (df.reset_index('device_id').groupby('device_id')
                        .rolling(window_value, min_periods=1)
                        .agg({'daily_distance':'var'})
                        )
        win_df_var.columns = [win_df_var.columns.values + "_" + window_name + '_var']
        win_df_var = win_df_var.rename(
            columns = {"{0}_count_{1}".format(groupby_key, window_name) :
                       "count_{0}".format(window_name)
                       })

        # MAX
        win_df_max = (df.reset_index('device_id').groupby('device_id')
                        .rolling(window_value, min_periods=1)
                        .agg({'daily_distance':'max'})
                        )
        win_df_max.columns = [win_df_max.columns.values + "_" + window_name + '_max']
        win_df_max = win_df_max.rename(
            columns = {"{0}_count_{1}".format(groupby_key, window_name) :
                       "count_{0}".format(window_name)
                       })

        # COUNT
        win_df_count = (df.reset_index('device_id').groupby('device_id')
                        .rolling(window_value, min_periods=1)
                        .agg({'daily_distance':'count'})
                        )
        win_df_count.columns = [win_df_count.columns.values + "_" + window_name + '_count']
        win_df_count = win_df_count.rename(
            columns = {"{0}_count_{1}".format(groupby_key, window_name) :
                       "count_{0}".format(window_name)
                       })

        win_df = win_df_mean.join(win_df_var).join(win_df_max).join(win_df_count)
        if rolled_df.empty:
            rolled_df = win_df
        else:
            rolled_df = rolled_df.join(win_df, how='outer')

    return rolled_df
