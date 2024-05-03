import polars as pl
import pandas as pd
from glob import glob


def merge_n_case_ids(
    n_ids: int = 0, 
    data_dir: str = '../data/processed/grouped/new_aggs/',
    path_to_base: str = '../data/raw/csv_files/train/train_base.csv',
    use_0: bool = True,
    as_pandas: bool = False,
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
    for i, path in enumerate(file_paths):
        if i == 0:
            df = pl.read_parquet(path)
            df = df.filter(pl.col('case_id').is_in(case_ids))
        else:
            temp = pl.read_parquet(path)
            temp = temp.filter(pl.col('case_id').is_in(case_ids))
            df = df.join(temp, on='case_id', how='outer_coalesce')
    del temp

    if as_pandas:
        df = df.to_pandas()

    return df