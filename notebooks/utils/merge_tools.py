import polars as pl
import pandas as pd
from glob import glob


def merge_n_case_ids(
    n_ids: int = 0, 
    data_dir: str = '../data/processed/grouped/new_aggs/',
    path_to_base: str = '../data/raw/csv_files/train/train_base.csv',
    use_0: bool = True,
    as_pandas: bool = False,
    case_id_list: list = [],
    random_state: int = 28
) -> pl.DataFrame | pd.DataFrame:
    '''
    Function to merge all parquet files, can return subset case_id.

    Parameters
    ----------
    n_ids : Number of case_ids to keep (int)
    data_dir : Path to processed parquet files directory (str)
    path_to_base : Path to base file (str)
    use_0 : Use num_group1 == 0 (bool)
    as_pandas : Return as pandas DataFrame
    case_id_list : List of case_ids to retrieve (list)
    random_seed : Random seed (int)
    '''
    # Get random sample of case_ids
    base_df = pd.read_csv(path_to_base)
    if n_ids > 0:
        if len(case_id_list) == 0:
            weights = pd.Series(1, index=base_df.index)
            target_column = 'target'
            target_weight = 5
            weights[base_df.index[base_df[target_column] == 1]] = target_weight
            case_ids = base_df.sample(n=n_ids, replace=False, weights=weights, random_state=random_state)
            case_ids = list(case_ids['case_id'])
        else:
            assert n_ids == len(case_id_list), 'length of case_id_list must equal n_ids'
            case_ids = case_id_list
    else:
        case_ids = list(base_df['case_id'])
    del base_df

    # Get files
    if use_0:
        file_paths = glob(data_dir + '*grouped_0.parquet')
    else:
        file_paths = glob(data_dir + '*grouped_rest.parquet')

    # Merge DataFrames
    df = pl.read_csv(path_to_base).filter(pl.col('case_id').is_in(case_ids))
    for path in file_paths:
        temp = pl.read_parquet(path)
        temp = temp.filter(pl.col('case_id').is_in(case_ids))
        df = df.join(temp, on='case_id', how='outer_coalesce')
    del temp

    if as_pandas:
        df = df.to_pandas()

    return df


def merge_n_case_ids_batch_processing(
    n_ids: int = 0, 
    data_dir: str = '../data/processed/grouped/new_aggs/',
    path_to_base: str = '../data/raw/csv_files/train/train_base.csv',
    use_0: bool = True,
    as_pandas: bool = False,
    batch_size: int = 15,
    random_state: int = 42
    
) -> pl.DataFrame | pd.DataFrame:
    '''
    Function to merge all parquet files, can return subset case_id.

    Parameters
    ----------
    n_ids : Number of case_ids to keep (int)
    data_dir : Path to processed parquet files directory (str)
    path_to_base : Path to base file (str)
    use_0 : Use num_group1 == 0 (bool)
    as_pandas : Return as pandas DataFrame
    random_seed : Random seed (int)
    '''
    # Get random sample of case_ids
    base_df = pd.read_csv(path_to_base)
    if n_ids > 0:
        case_ids = list(base_df['case_id'].sample(n=n_ids, replace=False, random_state=random_state))
    else:
        case_ids = list(base_df['case_id'])
    del base_df

    # Get files
    if use_0:
        file_paths = glob(data_dir + '*grouped_0.parquet')
    else:
        file_paths = glob(data_dir + '*grouped_rest.parquet')

    # Merge DataFrames
    df = pl.read_csv(path_to_base)

    # Process in batches
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i+batch_size]
        batch_dfs = [pl.read_parquet(path) for path in batch_paths]
        for temp in batch_dfs:
            temp = temp.filter(pl.col('case_id').is_in(case_ids))
            df = df.join(temp, on='case_id', how='outer_coalesce')
        del batch_dfs

    if as_pandas:
        df = df.to_pandas()

    return df