import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

log_dir = 'logs'
os.makedirs(log_dir,exist_ok = True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

customHandler = logging.StreamHandler()
customHandler.setLevel('DEBUG')

filepath = os.path.join(log_dir,'data_ingestion.log')
fileHandler = logging.FileHandler(filepath)
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
customHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(customHandler)

def load_data(csv_file:str)-> pd.DataFrame:
    """ load data from csv file """
    try:
        df = pd.read_csv(csv_file)
        logger.debug('file loaded suucessfully')
        return df
    
    except Exception as e:
        logger.error('File not loaded: %s', e)

def preprocessing_data(data:pd.DataFrame)-> pd.DataFrame:
    """Remove unwanted columns and do some basic prepration of data"""
    try:
        data = data.drop(['id','location','keyword'] , axis = 1)
        logger.debug(' preprocessing done')
        return data
    except Exception as e:
        logger.error("Error : ",e)


def save_data(train_data:pd.DataFrame,val_data:pd.DataFrame,data_dir:str) -> None:
    """ save the train and validation data """ 
    try:
        raw_data_path = os.path.join(data_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_file_path = os.path.join(raw_data_path,'train.csv')
        val_file_path = os.path.join(raw_data_path,'val.csv')
        train_data.to_csv(train_file_path)
        val_data.to_csv(val_file_path)
        logger.debug('File saved succesfully')
    except Exception as e:
        logger.error(' there is some error in saving data: ',e)




               



def main():
    data = load_data('./train_data.csv')
    data = preprocessing_data(data)
    test_size = 0.2
    X_train,X_val = train_test_split(data, test_size=0.2, random_state=22)
    print('train shape : ', X_train.shape)
    print('validation data shape: ', X_val.shape)
    save_data(X_train,X_val,'./data')
    

if __name__ == '__main__':
    main()
            


