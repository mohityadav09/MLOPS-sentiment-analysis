import os
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml


log_dir = 'logs'
os.makedirs(log_dir,exist_ok = True)

logger = logging.getLogger('feature-engineering')
logger.setLevel('DEBUG')

customHandler = logging.StreamHandler()
customHandler.setLevel('DEBUG')

filepath = os.path.join(log_dir,'feature_eng.log')
fileHandler = logging.FileHandler(filepath)
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
customHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(customHandler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->tuple:
    """apply tfidf to data"""
    try:
        tfidf = TfidfVectorizer(max_features=max_features)
        X_train = train_data['text'].values
        
        X_test = test_data['text'].values
        
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['target'] = train_data['target'].values
        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['target'] = test_data['target'].values

        logger.debug('tfidf applied')
        return train_df,test_df
    
    except Exception as e:
        logger.error('vectorization failed %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        train_data = pd.read_csv('./data/interim/train_processed.csv')
        test_data =  pd.read_csv('./data/interim/test_processed.csv')
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        logger.debug('data loaded succesfully')
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        train_df,test_df = apply_tfidf(train_data,test_data,max_features)
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()






