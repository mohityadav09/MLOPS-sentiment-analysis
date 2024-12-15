import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

import logging
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
nltk.download('punkt')

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)
logger = logging.getLogger('data_processing')
logger.setLevel('DEBUG')

customHandler = logging.StreamHandler()
customHandler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

customHandler.setFormatter(formatter)

logger.addHandler(customHandler)


def transform_text(text):
    """ Basic preprocessing on text such as removing stopwords , stemming the word removing links """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)
    
def preprocess_df(data:pd.DataFrame,target_column='target',text_column='text') -> pd.DataFrame:
    """Preprocesses the DataFrame by encoding the target column, removing duplicates, and 
    transforming the text column."""  
    try:
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        logger.debug('label encoding succesful')
        data.drop_duplicates(keep='first')
        logger.debug('removed duplicates')
        data.dropna(inplace=True)
        logger.debug('drop null values')
    
        data[text_column] = data[text_column].apply(transform_text)

        logger.debug('data processing done')
        return data

    except Exception as e:
        logger.error('there is error in preprocessing ', e)
        raise


def main():
     
     try:
         train_data = pd.read_csv('./data/raw/train.csv')
         test_data = pd.read_csv('./data/raw/val.csv')

         logger.debug('data loaded successfully')

         train_data_preprocessed = preprocess_df(train_data)
         test_data_preprocessed = preprocess_df(test_data)

         data_path = os.path.join("./data", "interim")
         os.makedirs(data_path, exist_ok=True)

         train_data_preprocessed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
         test_data_preprocessed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
         logger.debug('Processed data saved to %s', data_path)

     except Exception as e:
         logger.error('error in preprocessing data ',e)
         raise
     
if __name__ == '__main__':
    main() 

     

          

    



