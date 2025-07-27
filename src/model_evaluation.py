import os
import pandas as pd
import logging
import joblib
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s -%(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info('Model Evaluation Started')

def load_model(model_path: str):
    try:
        with open(model_path, "rb") as model_file:
            model = joblib.load(model_file)
            return model
    except Exception as e:
        raise e
    except FileNotFoundError as e:
        raise e
    

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


def evaluate_model(clf, X_test:pd.DataFrame, y_test:pd.DataFrame) -> dict:
    """model evaluating with statistical methods"""
    try:
        logger.info('Model Evaluation Started')
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict =  {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }
        logging.info("model evaluation metrics calculated")
        return metrics_dict
    
    except Exception as e:
        logging.error("error occurs in model evaluation %s",e)
        raise



def save_metrics(metrics:dict,file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info("model evaluation metrics saved to %s",file_path)

    except Exception as e:
        logging.error("error occurs in saving the model evaluation metrics %s",e)
        raise


def main():
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, x_test, y_test)
        save_metrics(metrics, 'reports/metrics.json')


    except Exception as e:
        logging.error("error occurs in model evaluation %s",e)
        raise


if __name__=="__main__":
    main()

