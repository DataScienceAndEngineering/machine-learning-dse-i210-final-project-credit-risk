import polars as pl
import glob
import os

import utils.merge as M

BASE_DIR = 'data/raw/parquet_files/train/'
FILE_GROUPS = [
    # 'train_static_0', 'train_static_cb', 
    'train_tax_registry_a', 'train_tax_registry_b', 'train_tax_registry_c',
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
            temp_df = pl.read_parquet(file).pipe(M.set_table_dtypes)
            df = pl.concat([df, temp_df])

        # Date columns
        date_cols = [ df.columns[i] for i in range(len(df.columns)) if df.dtypes[i] == pl.Date ]

        # Categorical columns
        cat_cols = [ 
            df.columns[i] for i in range(len(df.columns)) 
            if df.dtypes[i] == pl.String
        ]

        # Numerical columns
        ignore_cols = ['case_id', 'num_group1', 'num_group2']
        num_cols = [ 
            df.columns[i] for i in range(len(df.columns)) 
            if (df.columns[i] not in date_cols) and (df.columns[i] not in cat_cols) and (df.columns[i] not in ignore_cols)
        ]

        # Extract year, month, day
        # date_df = M.separate_dates(df, date_cols)

        # Frequency encode cat columns
        # freq_df = M.freq_encoder(df, cat_cols)

        # Binary encode cat columns
        # bin_df = M.binary_encoder(df, cat_cols)

        # Create and write DataFrames
        if 'num_group1' in df.columns:
            df_num_group_0 = df.filter(pl.col('num_group1') == 0)
            df_num_group_rest = df.filter(pl.col('num_group1') != 0)

            df_num_group_0 = M.group_file_data(df_num_group_0, num_cols, date_cols, cat_cols)
            df_num_group_rest = M.group_file_data(df_num_group_rest, num_cols, date_cols, cat_cols)

            new_date_cols = [ 
                col for col in df_num_group_0.columns 
                if df_num_group_0[col].dtype == pl.Date 
            ]

            # Transform data
            date_df_0 = M.separate_dates(df_num_group_0, new_date_cols)
            date_df_rest = M.separate_dates(df_num_group_rest, new_date_cols)
            freq_df_0 = M.freq_encoder(df_num_group_0, cat_cols)
            freq_df_rest = M.freq_encoder(df_num_group_rest, cat_cols)
            bin_df_0 = M.binary_encoder(df_num_group_0, cat_cols)
            bin_df_rest = M.binary_encoder(df_num_group_rest, cat_cols)

            # Merge data
            df_0 = pl.concat([df_num_group_0, date_df_0, freq_df_0, bin_df_0], how='horizontal')
            df_rest = pl.concat([df_num_group_rest, date_df_rest, freq_df_rest, bin_df_rest], how='horizontal')

            # Drop unprocessed columns
            df_0 = df_0.drop(date_cols + cat_cols)
            df_rest = df_rest.drop(date_cols + cat_cols)

            output_file_0 = df_0.write_parquet(os.path.join(OUTPUT_DIR, file_group + '_grouped_0.parquet'))
            output_file_rest = df_rest.write_parquet(os.path.join(OUTPUT_DIR, file_group + '_grouped_rest.parquet'))
        else:
            df = M.group_file_data(df, num_cols, date_cols)
            output_file = df.write_parquet(os.path.join(OUTPUT_DIR, file_group + '_grouped.parquet'))