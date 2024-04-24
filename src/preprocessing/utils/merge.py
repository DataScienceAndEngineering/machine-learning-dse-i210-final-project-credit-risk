import polars as pl
from sklearn.preprocessing import LabelEncoder

def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    # implement here all desired dtypes for tables
    # the following is just an example
    for col in df.columns:
        # last letter of column name will help you determine the type
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))

        if col[-1] == 'D':
            df = df.with_columns(pl.col(col).str.to_date())

    return df

def separate_dates(df: pl.DataFrame, date_cols: list[str] = []) -> pl.DataFrame:

    date_df = df.select(date_cols)

    for col in date_cols:
        date_df = date_df.with_columns(pl.col(col).dt.year().alias(col + '_year'))
        date_df = date_df.with_columns(pl.col(col).dt.month().alias(col + '_month'))
        date_df = date_df.with_columns(pl.col(col).dt.day().alias(col + '_day'))
        date_df.drop_in_place(col)

    return date_df

def freq_encoder(df: pl.DataFrame, cat_cols: list[str] = []) -> pl.DataFrame:

    cat_df = df.select(cat_cols)

    for col in cat_cols:
        # Calculate frequency for each category in the column
        value_counts = cat_df.groupby(col).agg(pl.len().alias('count'))
        total_count = cat_df.height  # Use height for row count in Polars
        frequency = (value_counts.with_columns(
                        (value_counts['count'] / total_count).alias(f'{col}_freq')
                    ).select([col, f'{col}_freq']))
        
        # Joining the frequency DataFrame back to the original DataFrame
        cat_df = cat_df.join(frequency, on=col, how='left')
        cat_df.drop_in_place(col)

    return cat_df

def binary_encoder(df: pl.DataFrame, cat_cols: list[str] = []) -> pl.DataFrame:
    # Initialize LabelEncoder
    le = LabelEncoder()

    cat_df = df.select(cat_cols)

    # Binary encoding for each categorical column
    for col in cat_cols:
        # Convert categories to integers using LabelEncoder from sklearn
        encoded_int = le.fit_transform(cat_df[col].to_numpy())

        # Convert the numpy array back to a Polars Series and rename it
        int_col_name = f"{col}_int"
        encoded_series = pl.Series(encoded_int).alias(int_col_name)

        # Add the integer encoded column to the DataFrame
        cat_df = cat_df.with_columns(encoded_series)

        # Calculate the maximum binary length
        max_binary_length = encoded_series.max().bit_length()

        # Create binary encoding directly
        for bit_position in range(max_binary_length):
            # Use bitwise operations directly within Polars
            bit_value = (encoded_series / (2 ** bit_position)).cast(pl.Int64) & 1
            new_col_name = f"{col}_binary_{bit_position}"
            cat_df = cat_df.with_columns(
                bit_value.alias(new_col_name)
            )
            cat_df = cat_df.with_columns(
                pl.col(new_col_name).cast(pl.Int8)
            )

        cat_df.drop_in_place(int_col_name)
        cat_df.drop_in_place(col)
    
    return cat_df

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
    cat_cols : List of categorical column names
    '''
    
    # Convert date columns
    df_date = df[['case_id'] + date_cols]

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

    # Cat aggs
    # cat_aggs = [ pl.col(col).mode().name.suffix('_mode') for col in cat_cols ] +\
            #    [ pl.n_unique(col).name.suffix('_distinct') for col in cat_cols ]
    cat_aggs = [ pl.col(col).mode() for col in cat_cols ]
    df_cat_grouped = df_cat.group_by('case_id').agg(cat_aggs)
    for col in df_cat_grouped.columns:
        if df_cat_grouped[col].dtype != pl.Int64:
            df_cat_grouped = df_cat_grouped.with_columns(pl.col(col).map_elements(lambda x: x[0], return_dtype=pl.String))

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