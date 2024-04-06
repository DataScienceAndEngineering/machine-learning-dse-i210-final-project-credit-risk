import polars as pl

def group_file_data(
    df: pl.DataFrame, 
    num_cols: list[str] = [], 
    date_cols: list[str] = [], 
    cat_cols: list[str] = []
) -> pl.DataFrame:
    '''
    Function to group numerical, date, and categorical columns

    Parameters:
    -----------
    df : Polars DataFrame
    num_cols : List of numerical column names (remember to drop num_group columns)
    date_cols : List of date column names
    cat_cols : List of categorical column names (becomes dummies)
    '''
    
    # Convert date columns
    df_date = df[['case_id'] + date_cols].with_columns([ pl.col(col).str.to_date() for col in date_cols ])

    # One-hot categories
    # df_dummies = df[['case_id'] + cat_cols].to_dummies(cat_cols)
    df_cat = df[['case_id'] + cat_cols]

    # Num DataFrame
    df_num = df[['case_id'] + num_cols]

    # Date aggs
    date_aggs = [ pl.min(col).name.suffix('_min') for col in date_cols ] +\
                [ pl.max(col).name.suffix('_max') for col in date_cols ] +\
                [ pl.n_unique(col).name.suffix('_distinct') for col in date_cols]
    df_date_grouped = df_date.group_by('case_id').agg(date_aggs)

    # One-hot aggs
    # dummy_cols = [ col for col in df_dummies.columns if col != 'case_id']
    # dummies_aggs = [ pl.sum(col).name.suffix('_sum') for col in dummy_cols ]
    # df_dummies_grouped = df_dummies.group_by('case_id').agg(dummies_aggs)
    cat_aggs = [ pl.col(col).mode().name.suffix('_mode') for col in cat_cols ] +\
               [ pl.n_unique(col).name.suffix('_distinct') for col in cat_cols ]
    df_cat_grouped = df_cat.group_by('case_id').agg(cat_aggs)

    # Numerical aggs
    num_aggs = [ pl.min(col).name.suffix('_min') for col in num_cols ] +\
            [ pl.max(col).name.suffix('_max') for col in num_cols ] +\
            [ pl.mean(col).name.suffix('_mean') for col in num_cols ] +\
            [ pl.median(col).name.suffix('_median') for col in num_cols ] +\
            [ pl.sum(col).name.suffix('_sum') for col in num_cols ]
    df_num_grouped = df_num.group_by('case_id').agg(num_aggs)

    # Join DataFrames
    df_joined = df_num_grouped.join(df_date_grouped, on='case_id')
    # df_joined = df_joined.join(df_dummies_grouped, on='case_id')
    df_joined = df_joined.join(df_cat_grouped, on='case_id')

    return df_joined