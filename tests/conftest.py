'''
Author: Vitor Abdo

This .py file is for creating the fixtures
'''
import pytest
from decouple import config
from customer_churn_model.etl import import_data, transform_data

# config
NEW_DATA = config('NEW_DATA')
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')


@pytest.fixture()
def sample_input_data():
    '''Fixture to generate the raw data'''
    raw_df = import_data(NEW_DATA)
    return raw_df


@pytest.fixture()
def sample_transformed_data():
    '''Fixture to generate transformed etl data'''
    raw_df = import_data(NEW_DATA)
    df_transformed = transform_data(raw_df)
    return df_transformed


@pytest.fixture()
def sample_input_X():
    '''Fixture to generate independent features (X)'''
    raw_df = import_data(NEW_DATA)
    df_transformed = transform_data(raw_df)
    X = df_transformed.drop([TARGET_AFTER_ETL], axis=1)
    return X
