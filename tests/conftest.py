'''
Author: Vitor Abdo

This .py file is for creating the fixtures
'''

import pytest
from decouple import config
from etl_workflow.etl import import_data, transform_data

# config
DATASET = config('DATASET')
TARGET_BEFORE_ETL = config('TARGET_BEFORE_ETL')


@pytest.fixture()
def sample_input_data():
    raw_df = import_data(DATASET)
    return raw_df


@pytest.fixture()
def sample_transformed_data():
    raw_df = import_data(DATASET)
    df_transformed = transform_data(raw_df)
    return df_transformed


@pytest.fixture()
def sample_input_X():
    raw_df = import_data(DATASET)
    df_transformed = transform_data(raw_df)
    X = df_transformed.drop([TARGET_BEFORE_ETL], axis=1)
    return X
