import polars as pl
import glob
import os

from utils.merge import group_file_data

BASE_DIR = 'data/raw/parquet_files/train/'
FILE_GROUPS = [
    'train_static_0', 'train_static_cb', 'train_tax_registry_a',
    'train_tax_registry_b', 'train_tax_registry_c',
    'train_applprev_1', 'train_applprev_2',
    'train_credit_bureau_a_1', 'train_credit_bureau_a_2',
    'train_credit_bureau_b_1', 'train_credit_bureau_b_2',
    'train_static_0', 'train_static_cb', 'train_tax_registry_a',
    'train_tax_registry_b', 'train_tax_registry_c',
    'train_applprev_1', 'train_applprev_2',
    'train_credit_bureau_a_1', 'train_credit_bureau_a_2',
    'train_credit_bureau_b_1', 'train_credit_bureau_b_2',
    'train_debitcard_1', 'train_deposit_1', 'train_other_1',
    'train_person_1', 'train_person_2'
]
OUTPUT_DIR = 'data/processed/grouped/new_aggs/'

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for file_group in FILE_GROUPS:
        print(f'Processing "{file_group}" files...')
        files = glob.glob(os.path.join(BASE_DIR, file_group) + '*')
    
        # Get columns
        df = pl.DataFrame()
        for file in files:
            temp_df = pl.read_parquet(file)
            df = pl.concat([df, temp_df])

        # Date columns
        date_cols = [ df.columns[i] for i in range(len(df.columns)) if (df.columns[i].__contains__('dat')) and (df.dtypes[i] == pl.String) ]

        # Categorical columns
        cat_cols = [ df.columns[i] for i in range(len(df.columns)) if (df.columns[i] not in date_cols) and (df.dtypes[i] == pl.String) ]

        # Numerical columns
        ignore_cols = ['case_id', 'num_group1', 'num_group2']
        num_cols = [ 
            df.columns[i] for i in range(len(df.columns)) 
            if (df.columns[i] not in date_cols) and (df.columns[i] not in cat_cols) and (df.columns[i] not in ignore_cols)
        ]

        # Drop cat cols
        df = df.drop(cat_cols)

        # Create and write DataFrames
        if 'num_group1' in df.columns:
            df_num_group_0 = df.filter(pl.col('num_group1') == 0)
            df_num_group_rest = df.filter(pl.col('num_group1') != 0)

            df_num_group_0 = group_file_data(df_num_group_0, num_cols, date_cols)
            df_num_group_rest = group_file_data(df_num_group_rest, num_cols, date_cols)

            output_file_0 = df_num_group_0.write_parquet(os.path.join(OUTPUT_DIR, file_group + '_grouped_0.parquet'))
            output_file_rest = df_num_group_rest.write_parquet(os.path.join(OUTPUT_DIR, file_group + '_grouped_rest.parquet'))
        else:
            df = group_file_data(df, num_cols, date_cols)
            output_file = df.write_parquet(os.path.join(OUTPUT_DIR, file_group + '_grouped.parquet'))