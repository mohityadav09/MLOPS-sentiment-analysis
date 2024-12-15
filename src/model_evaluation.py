import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
from dvclive import Live
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

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



def load_data(filepath: str):
    """
    Load test data from a processed data file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        tuple: Features (X_test) and labels (label) as NumPy arrays.

    Raises:
        Exception: If data loading fails.
    """
    try:
        # Attempt to read with multiple encodings
        encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1']
        for encoding in encodings_to_try:
            try:
                test_df = pd.read_csv(filepath, encoding=encoding)
                logger.debug("Data loaded successfully using encoding: %s", encoding)
                break
            except UnicodeDecodeError:
                logger.warning("Failed to read file with encoding: %s. Retrying...", encoding)
        else:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode file with tried encodings.")
        
        # Ensure the 'target' column exists
        if 'target' not in test_df.columns:
            raise KeyError("The 'target' column is missing in the provided file.")
        
        # Separate features and labels
        X_test = test_df.drop('target', axis=1)
        label = test_df['target']
        
        logger.debug('Test data loaded successfully from filepath: %s', filepath)
        return X_test.values, label.values
    
    except FileNotFoundError as fnf_error:
        logger.error('File not found: %s', fnf_error)
        raise
    except KeyError as key_error:
        logger.error('Key error: %s', key_error)
        raise
    except Exception as e:
        logger.error('Data loading failed: %s', e)
        raise

def load_model(model_path: str):
    """
    Load the machine learning model for prediction.

    Args:
        model_path (str): Path to the model file.

    Returns:
        The loaded model.

    Raises:
        Exception: If model loading fails.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded successfully from %s', model_path)
        return model
    except FileNotFoundError as fnf_error:
        logger.error('Model file not found: %s', fnf_error)
        raise
    except Exception as e:
        logger.error('Model loading failed: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model and return the evaluation metrics.

    Args:
        clf: The trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels.

    Returns:
        dict: Evaluation metrics.
    """
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise 

def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save the evaluation metrics to a JSON file.

    Args:
        metrics (dict): Evaluation metrics.
        file_path (str): Path to save the metrics.

    Raises:
        Exception: If saving metrics fails.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    """
    Main function to orchestrate data loading, model evaluation, and saving metrics.
    """
    try:
        data_path = './data/processed/test_tfidf.csv'
        model_path = './models/model.pkl'
        metrics_path = './reports/metrics.json'
        
        # Load data
        X_test, y_test = load_data(data_path)

        #load params
        params = load_params('params.yaml')
        
        # Load model
        model = load_model(model_path)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', metrics['accuracy'])
            live.log_metric('precision', metrics['precision'])
            live.log_metric('recall', metrics['recall'])

            live.log_params(params)
        
        # Save metrics
        save_metrics(metrics, metrics_path)
        logger.info('Model evaluation process completed successfully')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
