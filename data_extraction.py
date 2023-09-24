from kaggle.api.kaggle_api_extended import KaggleApi
import os

# Initializing Kaggle API client and authenticating
api = KaggleApi()
api.authenticate()

# Specifying the dataset name for download
dataset_name = 'ahmedelsayedrashad/airline-on-time-performance-data'

# Specifying destination folder to save the dataset
destination_folder = '/Users/khushal/Desktop/Projects/XGBoost_ETL_Pipeline/Dataset'

# UN-ZIP 
api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)
