import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
import yaml


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s -%(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info('Data Ingestion Started')

logger.info('Data Ingestion Completed')

logger.info('Data Ingestion Completed')


def load_params(params_path: str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logging.debug("parameters received from yaml file are: %s", params_path)
        return params
    
    except Exception as e:
        logging.error(f"Error loading parameters from {params_path}: {e}")
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """load data from csv file"""

    try:
        df=pd.read_csv(data_url)
        logging.debug(f"Data loaded from {data_url}")
        return df
    
    except pd.errors.ParserError as e:
        logging.error(f"Error loading data from {data_url}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data from {data_url}: {e}")
        raise


def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """preprocess data"""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns = {'v1':'target','v2': 'text'}, inplace=True)
        logging.debug("Data preprocessing completed")
        return df
    
    except KeyError as e:
        logging.error(f"Missing Columns in dataframe: {e}")
        raise
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise


def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path: str) -> None:
    """save data to train and split"""
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logging.debug("Data saved to csv file %s", raw_data_path)
    except Exception as e:
        logging.error(f"Error saving data to csv file: {e}")
        raise



def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        #test_size=0.2
        data_path='https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')

    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise


if __name__ == '__main__':
    main()   