import os
import zipfile
import kaggle

def download_and_unzip_dataset(competition, path='./data'):
    """
    Downloads a dataset from a Kaggle competition and unzips it.

    Parameters:
    competition (str): The Kaggle competition name.
    path (str): The local directory where the dataset should be saved.
    """
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    # Use Kaggle API to download the dataset
    kaggle.api.competition_download_files(competition, path=path, quiet=False)

    # Unzip all downloaded files
    for file in os.listdir(path):
        if file.endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as zip_ref:
                print(f"Unzipping {file}...")
                zip_ref.extractall(path)
                print(f"{file} unzipped successfully")

    print(f'Dataset downloaded and unzipped in {path}')

# Define my competition and download unzipped file
competition_name = 'home-credit-credit-risk-model-stability'
download_path = 'D:/'
download_and_unzip_dataset(competition_name, download_path)