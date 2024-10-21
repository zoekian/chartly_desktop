def get_avg_col(df: pd.DataFrame, target: str, date_col: str) -> pd.DataFrame:
    df_grouped = df.groupby([date_col]).agg({target: ['sum', 'count']})
    
    df_grouped.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df_grouped.columns]
    df_avg = df_grouped.agg({target + '_sum': 'cumsum', target + '_count': 'cumsum'})
    # return df_avg
    df_avg[f'avg_{target}'] = df_avg[target + '_sum'] / df_avg[target + '_count']
    df_avg.reset_index(inplace=True)
    df_avg = df_avg[['date', f'avg_{target}']]
    
    return clip_outliers_iqr(df_avg, [f'avg_{target}'])

def encode_smoothing(df: pd.DataFrame, feature_name: str, target: str, date_col: str, a: float = 100,
                     new_feature_name: str = None) -> pd.DataFrame:
    """
    Encode a feature using the formula: (Target(x) + a * avg_sales) / (Count(x) + a)
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    feature_name (str): The feature to encode (e.g., 'family', 'store_nbr').
    target (str): The column representing sales.
    date_col (str): The column respresenting dates.
    a (int): The smoothing parameter (regularization factor).
    new_feature_name (str): Name of the new feature.

    Returns:
    pd.DataFrame: The DataFrame with the encoded feature added.
    """
    if new_feature_name:
        enc_name = new_feature_name
    else:
        if a > 1:
            enc_name = f'{feature_name}_enc{int(a)}'
        else:
            enc_name = f'{feature_name}_enc{a}'
        
    if enc_name in df:
        df.drop([enc_name], axis=1, inplace=True)
    # Calculate the overall average target
    if f'avg_{target}' not in df:
        df = pd.merge(df, get_avg_col(df, target, date_col), on=date_col)

    
    df_gr = df.groupby(by=[feature_name, date_col]).agg({target: ['sum', 'count'], f'avg_{target}': 'max'})
    df_gr.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df_gr.columns]
    df_gr.rename(columns={target + '_sum': target, target + '_count': 'cnt', f'avg_{target}' + '_max': f'avg_{target}'}, inplace=True)

    df = df.sort_values(by=[feature_name, date_col])
    
    df_gr[[target, 'cnt']] = df_gr.groupby(feature_name).agg({target: 'cumsum', 'cnt': 'cumsum'})
    
    df_gr[enc_name] = (df_gr[target] + a * df_gr[f'avg_{target}']) / (df_gr['cnt'] + a)
    df_gr.reset_index(inplace=True)
    df = pd.merge(df, df_gr[[date_col, feature_name, enc_name]], on=[date_col, feature_name])
    return df

def encode_smoothing_by_list_of_values(df: pd.DataFrame, feature_names: list, target: str,
                                       date_col: str, a: float = 10,
                                      new_feature_name: str = None) -> pd.DataFrame:
    """
    Encode a feature using the formula: (Target(x1, x2, ..) + a * avg_sales) / (Count(x1, x2, ..) + a)
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    feature_name (list): The features to encode (e.g., ['family', 'store_nbr']).
    target (str): The column representing sales.
    date_col (str): The column respresenting dates.
    a (float): The smoothing parameter (regularization factor).
    new_feature_name (str): Name of the new feature. If name of the new feature in df, the function will drop the column.

    Returns:
    pd.DataFrame: The DataFrame with the encoded feature added.
    """
    if new_feature_name:
        enc_name = new_feature_name
    else:
        enc_name = ''
        for f in feature_names:
            enc_name += f + '_'
        if a > 1:
            enc_name += f'enc{int(a)}'
        else:
            enc_name += f'enc{a}'
    
    if enc_name in df:
        df.drop([enc_name], axis=1, inplace=True)
    # Calculate the overall average sales (avg_sales)
    if f'avg_{target}' not in df:
        df = pd.merge(df, get_avg_col(df, target, date_col), on=date_col)


    to_group = feature_names.copy()
    to_group.append(date_col)
    
    df_gr = df.groupby(by=to_group).agg({target: ['sum', 'count'], f'avg_{target}': 'max'})
    df_gr.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df_gr.columns]
    df_gr.rename(columns={target + '_sum': target, target + '_count': 'cnt', f'avg_{target}' + '_max': f'avg_{target}'}, inplace=True)

    df = df.sort_values(by=to_group)
    
    df_gr[[target, 'cnt']] = df_gr.groupby(feature_names).agg({target: 'cumsum', 'cnt': 'cumsum'})
    
    df_gr[enc_name] = (df_gr[target] + a * df_gr[f'avg_{target}']) / (df_gr['cnt'] + a)
    df_gr.reset_index(inplace=True)

    needed_cols = to_group.copy()
    needed_cols.append(enc_name)
    df = pd.merge(df, df_gr[needed_cols], on=to_group)
    return df

# Example usage:
# Assume df is your DataFrame with columns: 'family', 'sales'
# Apply smoothing encoding for the 'family' feature
