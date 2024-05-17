import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from glob import glob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE


def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    # implement here all desired dtypes for tables
    # the following is just an example
    for col in df.columns:
        # last letter of column name will help you determine the type
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))

        if col[-1] == 'D':
        # if col.__contains__('D'):
            try:
                df = df.with_columns(pl.col(col).str.to_date())
            except:
                pass
        
        if col == 'date_decision':
            df = df.with_columns(pl.col(col).str.to_date())


    return df

def load_data_group(group_name: str, base_file_path: str, data_dir: str) -> pl.DataFrame:
    '''
    Function to load all files from a single group.
    Note: Function removes non-target variable from merged base data.
    '''

    # Get path for all files in group
    paths = glob(os.path.join(data_dir, group_name) + '*')

    # Merge files into polars DataFrame
    df = pl.DataFrame()
    for path in paths:
        df = pl.concat([df, pl.read_parquet(path).pipe(set_table_dtypes)], how='vertical')

    base_df = pl.read_parquet(base_file_path).pipe(set_table_dtypes)
    df = df.join(base_df, on='case_id', how='left')

    return df

class WithinGroupImputer(BaseEstimator, TransformerMixin):
    def __init__(self, how: str, group_var: str=None):
        assert (how == 'mean') or (how == 'median'), 'Must be mean or median.'

        self.how = how
        self.group_var = group_var if group_var else 'group_var'
    
    def fit(self, X, y=None):
        self.imp_dict = {}

        if y is not None:
            self.group_vals = np.unique(y)
        else:
            assert self.group_var in X.columns, f"'{self.group_var}' not in columns."
            self.group_vals = X[self.group_var].unique()

        X_ = pd.concat([X, pd.Series(y, name=self.group_var)], axis=1)

        for col in X.columns:
            if pd.api.types.is_any_real_numeric_dtype(X_[col]):
                self.imp_dict[col] = {}
                if self.how == 'mean':
                    self.imp_dict[col]['default'] = X_[col].mean()
                    for val in self.group_vals:
                        self.imp_dict[col][val] = X_.groupby(self.group_var)[col].mean()[val]
                else:
                    self.imp_dict[col]['default'] = X_[col].median()
                    for val in self.group_vals:
                        self.imp_dict[col][val] = X_.groupby(self.group_var)[col].median()[val]
            
        return self
        
    def transform(self, X, y=None):
        if y is not None:
            X_ = pd.concat([X, pd.Series(y, name=self.group_var)], axis=1)
        else:
            X_ = X.copy()

        for col in X.columns:
            if pd.api.types.is_any_real_numeric_dtype(X_[col]):
                assert col in list(self.imp_dict.keys()), f"'{col}' is type {X_[col].dtypes}"
                
                for val in np.unique(y):
                    if val in self.group_vals:
                        X_.loc[X_[self.group_var] == val, col] = X_.loc[X_[self.group_var] == val, col].fillna(self.imp_dict[col][val])
                    else:
                        X_[col] = X_[col].fillna(self.imp_dict[col]['default'])

        return X_

def merge_n_case_ids(
    n_ids: int = 0, 
    data_dir: str = '../data/processed/grouped/new_aggs/',
    path_to_base: str = '../data/raw/csv_files/train/train_base.csv',
    # use_0: bool = True,
    target_weight: int = 5,
    as_pandas: bool = False,
    # case_id_list: list = [],
    random_state: int = 28
) -> pl.DataFrame | pd.DataFrame:
    '''
    Function to merge all parquet files, can return subset case_id.
    Test DataFrame from last 10000 cases, train DataFrame sampled from the
    remaining cases.

    Parameters
    ----------
    n_ids : Number of case_ids to keep (int)
    data_dir : Path to processed parquet files directory (str)
    path_to_base : Path to base file (str)
    use_0 : Use num_group1 == 0 (bool), NO LONGER IN USE
    target_weight : weighting for minority class sample (int)
    as_pandas : Return as pandas DataFrame
    case_id_list : List of case_ids to retrieve (list), NO LONGER IN USE
    random_seed : Random seed (int)

    Return
    ------
    train_df : Training DataFrame, sample from non-test cases
    test_df : Testing DataFrame, last 10000 cases
    '''
    # Get random sample of case_ids
    base_df = pd.read_csv(path_to_base)
    test_case_ids = base_df[-10000:]['case_id'].to_list()
    base_df = base_df[:-10000]
    
    if n_ids > 0:
        # if len(case_id_list) == 0:
        #     weights = pd.Series(1, index=base_df.index)
        #     target_column = 'target'
        #     target_weight = target_weight
        #     weights[base_df.index[base_df[target_column] == 1]] = target_weight
        #     case_ids = base_df.sample(n=n_ids, replace=False, weights=weights, random_state=random_state)
        #     case_ids = list(case_ids['case_id'])
        # else:
        #     assert n_ids == len(case_id_list), 'length of case_id_list must equal n_ids'
        #     case_ids = case_id_list
        weights = pd.Series(1, index=base_df.index)
        target_column = 'target'
        target_weight = target_weight
        weights[base_df.index[base_df[target_column] == 1]] = target_weight
        case_ids = base_df.sample(n=n_ids, replace=False, weights=weights, random_state=random_state)
        case_ids = list(case_ids['case_id'])
    else:
        case_ids = list(base_df['case_id'])
    del base_df

    # Get files
    # if use_0:
    #     file_paths = glob(data_dir + '*grouped_0.parquet')
    # else:
    #     file_paths = glob(data_dir + '*grouped_rest.parquet')

    # num_group1 == 0 paths
    file_paths = glob(data_dir + '*grouped_0.parquet')

    # num_group1 != 0 paths
    file_paths_rest = glob(data_dir + '*grouped_rest.parquet')

    # Merge DataFrames
    train_0_df = pl.read_csv(path_to_base).filter(pl.col('case_id').is_in(case_ids)).pipe(set_table_dtypes)
    test_0_df = pl.read_csv(path_to_base).filter(pl.col('case_id').is_in(test_case_ids)).pipe(set_table_dtypes)
    for path in file_paths:
        temp_df = pl.read_parquet(path).pipe(set_table_dtypes)
        train_cases = temp_df.filter(pl.col('case_id').is_in(case_ids))
        test_cases = temp_df.filter(pl.col('case_id').is_in(test_case_ids))
        train_0_df = train_0_df.join(train_cases, on='case_id', how='outer_coalesce')
        test_0_df = test_0_df.join(test_cases, on='case_id', how='outer_coalesce')

    del temp_df, train_cases, test_cases

    train_rest_df = pl.read_csv(path_to_base).filter(pl.col('case_id').is_in(case_ids)).pipe(set_table_dtypes)
    test_rest_df = pl.read_csv(path_to_base).filter(pl.col('case_id').is_in(test_case_ids)).pipe(set_table_dtypes)
    for path in file_paths_rest:
        temp_df = pl.read_parquet(path).pipe(set_table_dtypes)
        train_cases = temp_df.filter(pl.col('case_id').is_in(case_ids))
        test_cases = temp_df.filter(pl.col('case_id').is_in(test_case_ids))
        train_rest_df = train_rest_df.join(train_cases, on='case_id', how='outer_coalesce')
        test_rest_df = test_rest_df.join(test_cases, on='case_id', how='outer_coalesce')

    del temp_df, train_cases, test_cases

    train_df = train_0_df.join(train_rest_df, on='case_id', how='left')
    test_df = test_0_df.join(test_rest_df, on='case_id', how='left')
    del train_0_df, test_0_df, train_rest_df, test_rest_df

    if as_pandas:
        train_df = train_df.to_pandas()
        test_df = test_df.to_pandas()

    return train_df, test_df




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



def separate_dates(df: pl.DataFrame, date_cols: list[str] = []) -> pl.DataFrame:

    date_df = df.select(date_cols)

    for col in date_cols:
        date_df = date_df.with_columns(pl.col(col).dt.year().alias(col + '_year'))
        date_df = date_df.with_columns(pl.col(col).dt.month().alias(col + '_month'))
        date_df = date_df.with_columns(pl.col(col).dt.day().alias(col + '_day'))
        date_df.drop_in_place(col)

    return date_df

def create_is_null_cols(df: pl.DataFrame) -> pl.DataFrame:
    '''Only creates null columns for non-string columns'''
    for col in df.columns:
        if (df[col].is_null().sum() > 0) and (df[col].dtype != pl.String):
            df = df.with_columns(df[col].is_null().alias(f'{col}_is_null'))
    
    return df

def get_top_n_categories(df: pd.DataFrame, n_cat: int = 5) -> dict:
    cols = df.select_dtypes('object').columns
    cols_dict = {}
    for col in cols:
        if len(df[col].unique()) >= n_cat:
            top_n_list = df[col].value_counts()[:n_cat].index.tolist()
        else:
            top_n_list = df[col].unique().tolist()
        
        cols_dict[col] = top_n_list
        
    return cols_dict

def preprocess_and_tsne_with_nan_handling(X, y, title):
    # Copy the dataset to avoid changing the original
    X_encoded = X.copy()
    
    # Handle NaNs in categorical columns
    for column in X_encoded.select_dtypes(include=['object', 'category']).columns:
        X_encoded[column] = X_encoded[column].fillna('NULL')  # Replace NaNs with 'NULL'
        le = LabelEncoder()
        X_encoded[column] = le.fit_transform(X_encoded[column].astype(str))
    
    # Handle NaNs in numerical columns
    for column in X_encoded.select_dtypes(exclude=['object', 'category']).columns:
        X_encoded[column] = X_encoded[column].fillna(-999)  # Replace NaNs with -999
    
    # Initialize and apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X_encoded)
    
    # Create a DataFrame with t-SNE results and target
    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['target'] = y.reset_index(drop=True)
    
    # Plot the t-SNE results
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='target',
        palette=sns.color_palette('hsv', len(tsne_df['target'].unique())),
        data=tsne_df,
        alpha=0.6

    )
    plt.title(title)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend(title='Target')
    plt.grid(True)
    plt.show()

    print("t-SNE Visualization for Mean and Mode Imputed Validation Data with NaN Handling")

# Function to preprocess data, handle NaNs, and apply t-SNE
def tsne_plot(X, y, title):
    # Copy the dataset to avoid changing the original
    X_encoded = X.copy()
        
    # Initialize and apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X_encoded)
    
    # Create a DataFrame with t-SNE results and target
    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['target'] = y.reset_index(drop=True)
    
    # Plot the t-SNE results
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='target',
        palette=sns.color_palette('hsv', len(tsne_df['target'].unique())),
        data=tsne_df,
        alpha=0.6

    )
    plt.title(title)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend(title='Target')
    plt.grid(True)
    plt.show()

    print("t-SNE Visualization for Mean and Mode Imputed Validation Data with NaN Handling")