import logging
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s -%(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logging.info("starting the model training")

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logging.info("data loaded from %s file path of size %s",file_path,df.shape)
        return df
    
    except pd.errors.ParserError as e:
        logging.error("faied to parse the file%s",e)
        raise
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray, params:dict) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples")
        
        logging.info("initialing the random forest classifier with parameters %s",params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        clf.fit(X_train,y_train)
        logging.info("model trained successfully")
        return clf
    
    except Exception as e:
        logging.error(f"Error training the model: {e}")
        raise


def save_model(model,file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info("model saved to %s",file_path)

    except Exception as e:
        logging.error("error occurs in saving the model %s",e)
        raise

def main():
    try:
        params = {'n_estimators':25, 'random_state':2}
        trained_data = load_data('./data/processed/train_tfidf.csv')
        X_train = trained_data.iloc[:, :-1].values
        y_train = trained_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        save_model(clf,'models/model.pkl')
        logging.info("model saved successfully")

    except Exception as e:
        logging.error("error occurs %s",e) 
        raise


if __name__ == "__main__": 
    main()  







