import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir, "feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s -%(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info('Feature Engineering Started')

def load_data(file_path: str) -> pd.DataFrame:
    """load data from csv file"""

    try:
        df=pd.read_csv(file_path)
        logging.debug(f"Data loaded from {file_path}")
        df.fillna('', inplace=True)

        return df
    
    except pd.errors.ParserError as e:
        logging.error(f"failed to parse the file {file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise


def apply_tfidf(train_data:pd.DataFrame, test_data: pd.DataFrame, max_features:int) -> tuple:
    """apply tfidf vectorizer on train and test data"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        X_train = train_data['text'].values
        y_train = train_data['target'].values

        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        return train_df, test_df

    except Exception as e:
        logging.error(f"Error applying tfidf vectorizer: {e}")
        raise



def save_data(df:pd.DataFrame, file_path: str):
    """save data to csv file"""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.debug(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    try:
        max_features = 50

        train_data = load_data('./data/processed/interim/train_processed.csv')
        test_data = load_data('./data/processed/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

     

        save_data(train_df, os.path.join("./data","processed","train_tfidf.csv"))
        save_data(test_df, os.path.join("./data","processed","test_tfidf.csv"))

        logger.info('Feature Engineering Completed')

    except Exception as e:
        logging.error('failed to complete feature engineering %s', e)


if __name__=='__main__':
    main()        

       

        