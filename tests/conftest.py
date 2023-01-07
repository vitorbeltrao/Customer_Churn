'''
Author: Vitor Abdo

This .py file is for creating the fixtures
'''
import pytest
from decouple import config
from customer_churn_model.etl import import_data, transform_data

# config
DATASET = config('DATASET')
TEST_DATASET = config('TEST_DATASET')
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')
TARGET_BEFORE_ETL = config('TARGET_BEFORE_ETL')


@pytest.fixture()
def sample_input_data():
    '''Fixture to generate the raw data'''
    raw_df = import_data(DATASET)
    return raw_df


@pytest.fixture()
def sample_transformed_data():
    '''Fixture to generate transformed etl data'''
    raw_df = import_data(DATASET)
    df_transformed = transform_data(raw_df)
    return df_transformed


@pytest.fixture()
def sample_input_X():
    '''Fixture to generate independent features (X)'''
    raw_df = import_data(DATASET)
    df_transformed = transform_data(raw_df)
    X = df_transformed.drop([TARGET_AFTER_ETL], axis=1)
    return X


@pytest.fixture()
def sample_input_predict():
    '''Fixture to generate the input for predict function'''
    raw_df = import_data(TEST_DATASET)
    df_transformed = raw_df.drop([TARGET_AFTER_ETL], axis=1)
    return df_transformed.to_csv()
