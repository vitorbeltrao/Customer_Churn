'''
Author: Vitor Abdo

This .py file sequentially runs all the necessary tests for the functions created
for the whole system to work correctly
'''
import logging
import numpy
from decouple import config
from math import floor
from etl_workflow.etl import import_data, transform_data, split_dataset
from preprocessing import preprocessing


logging.basicConfig(
    filename='./logs/tests_system_funcs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# config
DATASET = config('DATASET')
TEST_SIZE = config('TEST_SIZE', cast=float)
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')


def test_import_data():
    '''tests the import_data function made in the etl.py file'''
    try:
        raw_df = import_data(DATASET)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError:
        logging.error("Testing import_data: The file wasn't found")

    try:
        assert raw_df.shape[0] > 0
        assert raw_df.shape[1] > 0
    except AssertionError:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")


def test_transform_data():
    '''tests the transform_data function made in the etl.py file'''
    try:
        raw_df = import_data(DATASET)
        df_transformed = transform_data(raw_df)
        logging.info("Testing transform_data: SUCCESS")
    except KeyError:
        logging.error(
            "Testing transform_data: Any variable subject to the transformations of this function was not found")

    try:
        assert df_transformed[TARGET_AFTER_ETL].dtypes == 'int64'
        assert 'Unnamed: 0' not in df_transformed.columns
        assert 'CLIENTNUM' not in df_transformed.columns
        assert 'Attrition_Flag' not in df_transformed.columns
    except AssertionError:
        logging.error(
            "Testing transform_data: The churn column didn't turn into 'int64' or the other columns weren't deleted correctly")


def test_split_dataset():
    '''tests the split_dataset function made in the etl.py file'''
    try:
        raw_df = import_data(DATASET)
        df_transformed = transform_data(raw_df)
        train_set, test_set = split_dataset(df_transformed)
        logging.info("Testing split_dataset: SUCCESS")
    except OSError:
        logging.error(
            "Testing split_dataset: Cannot save file into a non-existent directory")

    try:
        assert len(df_transformed) == len(train_set) + len(test_set)
        assert len(train_set) == floor(
            (1 - TEST_SIZE) * len(df_transformed))
        assert len(test_set) == floor(TEST_SIZE * len(df_transformed))
    except AssertionError:
        logging.error(
            "Testing split_dataset: The dataset was not split correctly")


def test_preprocessing():
    '''tests the preprocessing function made in the preprocessing.py file'''
    try:
        raw_df = import_data(DATASET)
        df_transformed = transform_data(raw_df)
        train_set, test_set = split_dataset(df_transformed)
        X_transformed = preprocessing(train_set)
        logging.info("Testing preprocessing: SUCCESS")
    except BaseException:
        logging.error(
            "Testing preprocessing: The function is not working correctly")

    try:
        assert X_transformed.shape[0] == (
            1 - TEST_SIZE) * transform_data(sample_input_data)
        assert X_transformed.shape[1] == 32
        assert isinstance(X_transformed, 'numpy.ndarray')
    except AssertionError:
        logging.error(
            "Testing preprocessing: The dataset was not pre-processed correctly")


if __name__ == "__main__":
    logging.info('About to start the tests')
    test_import_data()
    test_transform_data()
    test_split_dataset()
    test_preprocessing()
    logging.info('Done executing the tests')
