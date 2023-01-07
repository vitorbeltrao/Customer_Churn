'''
Author: Vitor Abdo

This .py file is for making the necessary transformations to feed
the machine learning algorithms using scikit-learn's column
transformer class
'''
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector


def preprocessing(independent_features: pd.DataFrame) -> ColumnTransformer:
    '''Functions to make the necessary transformations to feed
    the machine learning algorithm

    :param independent_features: (dataframe)
    The matrix of independent variables (X)

    :return:
    None
    '''
    # divide the qualitative and quantitative features
    try:
        quantitative_columns = selector(dtype_exclude=['object'])
        qualitative_columns = selector(dtype_include=['object'])

        quantitative_columns = quantitative_columns(independent_features)
        qualitative_columns = qualitative_columns(independent_features)

        # apply the respective transformations with columntransformer method
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), qualitative_columns)],
            remainder='passthrough')
        print("Execution of preprocessing: SUCCESS")
        return preprocessor
    except BaseException:
        print("Execution of preprocessing: FAILED")
        return None
