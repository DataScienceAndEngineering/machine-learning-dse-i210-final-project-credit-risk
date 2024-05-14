import pandas as pd
import polars as pl
import numpy as np
import random

def manual_balancing(base_file_path: str, random_state: int=28) -> pd.DataFrame:
    random.seed(random_state)

    df = pd.read_parquet(base_file_path)
    case_ids_1 = df.loc[df['target'] == 1]['case_id'].to_list()
    case_ids_2 = df.loc[df['target'] == 0]['case_id'].to_list()

    num_pos = len(case_ids_1)
    selection = random.sample(case_ids_2, num_pos)

    df = df.loc[(df['target'] == 1) | (df['case_id'].isin(selection))]

    return df