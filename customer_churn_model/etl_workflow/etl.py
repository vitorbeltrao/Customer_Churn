'''
Author: Vitor Abdo

This .py file is to do the ETL step to make possible the next
step that is the preprocessing and model training
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from decouple import config


# config
VARS_TO_DROP = config('VARS_TO_DROP')
TARGET_BEFORE_ETL = config('TARGET_BEFORE_ETL')
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')
TEST_SIZE = config('TEST_SIZE', cast=float)
SEED = config('SEED', cast=int)
NEW_TRAIN_DATA = config('NEW_TRAIN_DATA')
NEW_TEST_DATA = config('NEW_TEST_DATA')


def import_data(file_path: str) -> pd.DataFrame:
    '''Load dataset for the csv found at the path

    :param file_path: (str)
    A path to the csv

    :return: (dataframe)
    Pandas dataframe
    '''
    raw_df = pd.read_csv(file_path)
    return raw_df


def transform_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    '''Transform the raw dataset doing two steps of transformation.
    The first transformation is to transform the target variable in
    numeric. The second transformation is to drop some variables that
    we dont want to use in the model.

    :param df: (dataframe)
    The raw dataframe imported

    :return: (dataframe)
    Pandas dataframe transformed
    '''
    raw_df = import_data(raw_df)

    # transformations
    df_transformed = raw_df.copy()
    df_transformed[TARGET_AFTER_ETL] = raw_df[TARGET_BEFORE_ETL].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_transformed.drop([VARS_TO_DROP], axis=1, inplace=True)
    return df_transformed


def split_dataset(df_transformed: pd.DataFrame) -> pd.DataFrame:
    '''Function to split the transformed dataset in train and test.

    :param df_transformed: (dataframe)
    The transformed dataset

    :return:
    None
    '''
    train_set, test_set = train_test_split(transform_data(
        df_transformed), test_size=TEST_SIZE, random_state=SEED)
    train_set.to_csv(NEW_TRAIN_DATA, index=False)
    test_set.to_csv(NEW_TEST_DATA, index=False)
