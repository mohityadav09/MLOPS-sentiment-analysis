import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

def load_data(filepath:str)->pd.DataFrame:
    """Load data from csv"""
    try:
        df = pd.read_csv(filepath)
        logger.debug('data loaded succesfully')
        return df
    except Exception as e:
        logger.error('load data failes: %s',e)
        raise

def training_model(X_train:np.ndarray,label:np.ndarray,n_estimators=20,random_state=22) -> RandomForestClassifier:
    """Training of random forest model"""
    try:
        if X_train.shape[0] != label.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        clf = RandomForestClassifier(n_estimators=n_estimators,random_state= random_state)
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train,label)
        logger.debug('model training successful')
        return clf
    except Exception as e:
        logger.error('Training failed: %s ',e)

def save_model(model,filepath:str)->None:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath,'wb') as file:
            pickle.dump(model,file)

        logger.debug('Model saved to %s', filepath)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_filepath = './data/processed/train_tfidf.csv'
        train_df = load_data(train_filepath)
        X_train = train_df.drop('target',axis = 1).to_numpy()
        label = train_df['target'].to_numpy()
        params = load_params('params.yaml')
        model = training_model(X_train,label,params['model_building']['n_estimators'],params['model_building']['random_state'])

        model_dir ='models'
        os.makedirs(model_dir,exist_ok=True)
        model_save_path = 'models/model.pkl'
        save_model(model, model_save_path)
        logger.debug('succesfull')
    except Exception as e:
        logger.error('failed: %s ',e)  
        raise

if __name__ == '__main__':
    main()



