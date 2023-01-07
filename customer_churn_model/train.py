'''
Author: Vitor Abdo

This .py file is for training and saving the best model
'''
import warnings
import logging
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from decouple import config
from etl import import_data
from preprocessing import preprocessing
warnings.simplefilter(action='ignore')

logging.basicConfig(
    filename='./logs/logs_system_funcs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# config
CV_SCORING = config("CV_SCORING")
CV = config("CV", cast=int)
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')
SEED = config('SEED', cast=int)
NEW_TRAIN_DATA = config('NEW_TRAIN_DATA')
SAVE_PKL_FILE_NAME = config('SAVE_PKL_FILE_NAME')


def train_model() -> list:
    '''Function to train the model, tune the hyperparameters
    and save the best final model

    :return: (dataframe)
    Pandas dataframe
    '''
    # read training data
    df_train_transformed = import_data(NEW_TRAIN_DATA)

    # select only the features that we are going to use
    X = df_train_transformed.drop([TARGET_AFTER_ETL], axis=1)
    y = df_train_transformed[TARGET_AFTER_ETL]

    # apply the respective transformations with columntransformer method
    preprocessor = preprocessing(X)

    # Hyperparameter tunning
    # 1. Instantiate the pipeline
    final_model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('scaling', StandardScaler()),
            ('lgbm', LGBMClassifier(random_state=SEED))
        ]
    )

    # 2. Hyperparameter interval to be tested
    param_grid = {'lgbm__boosting_type': ['gbdt', 'rf'],
                  'lgbm__max_depth': [-1, 3, 5, 10, 15, 20, 50, 100],
                  'lgbm__learning_rate': [0.01, 0.05, 0.1, 0.5, 1]}

    # 3. Training and apply grid search with cross validation
    print('Starting to train the model with cross validation: ...')
    grid_search = GridSearchCV(
        final_model,
        param_grid,
        cv=CV,
        scoring=CV_SCORING,
        return_train_score=True)
    grid_search.fit(X, y)

    # 4. Instantiate best model
    final_model = grid_search.best_estimator_
    print('The best hyperparameters were:', grid_search.best_params_)

    cvres = grid_search.cv_results_
    for mean_train_score, mean_test_score in zip(cvres['mean_train_score'], cvres['mean_test_score']):
        print('Mean train score:', mean_train_score)
        print('Mean validation score:', mean_test_score)

    return joblib.dump(final_model, SAVE_PKL_FILE_NAME)


if __name__ == "__main__":
    logging.info('About to start the model train step of the system')
    train_model()
    logging.info('Done executing the model train step: SUCCESS')
