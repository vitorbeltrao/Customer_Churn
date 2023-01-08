'''
Author: Vitor Abdo

This .py file is for training and saving the best model
'''

import logging
import joblib
import math
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from decouple import config
from etl import import_data
from preprocessing import preprocessing

# config
CV_SCORING = config("CV_SCORING")
CV = config("CV", cast=int)
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')
SEED = config('SEED', cast=int)
TEST_DATASET = config('TEST_DATASET')
SAVE_PKL_FILE_NAME = config('SAVE_PKL_FILE_NAME')

# import the pkl file
_churn_pipeline = joblib.load(SAVE_PKL_FILE_NAME)


def test_model():
    '''
    '''
    # read testing data
    df_test_transformed = import_data(TEST_DATASET)

    # select only the features that we are going to use
    X = df_test_transformed.drop([TARGET_AFTER_ETL], axis=1)
    y = df_test_transformed[TARGET_AFTER_ETL]