'''
Author: Vitor Abdo

This .py file sequentially runs all the necessary tests for the functions created
for the whole system to work correctly
'''
from math import floor, ceil
from decouple import config
from customer_churn_model.etl import import_data, transform_data, split_dataset
from customer_churn_model.preprocessing import preprocessing
from customer_churn_model.predict import predict_churn

# config
DATASET = config('DATASET')
NEW_DATA = config('NEW_DATA')
TEST_SIZE = config('TEST_SIZE', cast=float)
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')


def test_import_data():
    '''tests the import_data function made in the etl.py file'''
    raw_df = import_data(DATASET)

    assert raw_df.shape[0] > 0
    assert raw_df.shape[1] > 0


def test_transform_data(sample_input_data):
    '''tests the transform_data function made in the etl.py file'''
    df_transformed = transform_data(sample_input_data)

    assert df_transformed[TARGET_AFTER_ETL].dtypes == 'int64'
    assert 'Unnamed: 0' not in df_transformed.columns
    assert 'CLIENTNUM' not in df_transformed.columns
    assert 'Attrition_Flag' not in df_transformed.columns


def test_split_dataset(sample_transformed_data):
    '''tests the split_dataset function made in the etl.py file'''
    train_set, test_set = split_dataset(sample_transformed_data)

    assert len(sample_transformed_data) == len(train_set) + len(test_set)
    assert len(train_set) == floor(
        (1 - TEST_SIZE) * len(sample_transformed_data))
    assert len(test_set) == ceil(TEST_SIZE * len(sample_transformed_data))


def test_preprocessing(sample_input_X):
    '''tests the preprocessing function made in the preprocessing.py file'''
    # Given
    preprocessor = preprocessing(sample_input_X)

    # When
    X_transformed = preprocessor.fit_transform(sample_input_X)

    # Then
    assert X_transformed.shape[0] == len(sample_input_X)
    assert X_transformed.shape[1] == 32


def test_predict_churn():
    '''tests the predict_churn function made in the predict.py file'''
    # Given
    results = predict_churn(NEW_DATA)
    lst = list(results.items())[0][1]

    # Then
    assert lst[:][0] == 0
    assert lst[:][5] == 0
    assert lst[:][14] == 0
    assert lst[:][1670] == 1
    assert lst[:][333] == 1
    assert isinstance(results, dict)
