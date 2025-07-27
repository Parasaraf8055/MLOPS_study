import os
from tracemalloc import stop
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir, "preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s -%(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info('Preprocessing Started')

def transform_text(text):
    """transforming the text """
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [word for word in text if word.isalnum()]

    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    text = [ps.stem(word) for word in text]

    return " ".join(text)

def pre_processing(df:pd.DataFrame, text_col='text', target_col='target'):
    """preprocessing the data by encoding the target column and transforming the text column"""

    try:
        logging.info("Preprocessing started")
        encoder = LabelEncoder()
        df[target_col] = encoder.fit_transform(df[target_col])
        logging.debug("Target column encoded")

        df = df.drop_duplicates(keep='first')
        logging.debug("Duplicates removed")

        df.loc[:, text_col] = df[text_col].apply(transform_text)
        logging.debug("Text transformed")
        return df
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise
    except Exception as e:
        logging.error(f"Error preprocessing data during normalization: {e}") 
        raise

def main():
    try:
        """main function to load data, transform it"""
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.debug("Data loaded from csv file")

        train_processed_data = pre_processing(train_data)
        test_processed_data = pre_processing(test_data)


        data_path = os.path.join('./data','interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logging.info("Data saved to csv file %s",data_path)

        logger.info('Preprocessing Completed')



    except FileNotFoundError as e:
        logging.error(f":file not found {e}")
        raise

    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

    except pd.errors.EmptyDataError as e:
        logging.error(f"No data: {e}")
        raise


if __name__ == '__main__':
    main()