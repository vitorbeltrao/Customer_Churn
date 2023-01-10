'''
Author: Vitor Abdo

This .py file is for training and saving the best model
'''
import warnings
import logging
import math
import joblib
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from decouple import config
from etl import import_data
from preprocessing import preprocessing
warnings.simplefilter(action='ignore')

# config
CV_SCORING = config("CV_SCORING")
CV = config("CV", cast=int)
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')
SEED = config('SEED', cast=int)
TRAIN_DATASET = config('TRAIN_DATASET')
SAVE_PKL_FILE_NAME = config('SAVE_PKL_FILE_NAME')
RESULT_IMAGES_PATH = config('RESULT_IMAGES_PATH')


def train_model() -> list:
    '''Function to train the model, tune the hyperparameters
    and save the best final model

    :return: (dataframe)
    Pandas dataframe
    '''
    try:
        # read training data
        df_train_transformed = import_data(TRAIN_DATASET)

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
        cvres = [(mean_test_score,
                  mean_train_score) for mean_test_score,
                 mean_train_score in sorted(zip(cvres['mean_test_score'],
                                                cvres['mean_train_score']),
                                            reverse=True) if (math.isnan(mean_test_score) != True)]
        print(
            'The mean test score and mean train score is, respectively:',
            cvres[0])

        logging.info('Execution of train model: SUCCESS')
        return final_model, joblib.dump(final_model, SAVE_PKL_FILE_NAME)

    except BaseException:
        logging.error('Execution of train model: FAILED')
        return None


def feature_importance_plot(model: Pipeline) -> plt.figure:
    '''Function to generate the graph of the
    most important variables for the model

    :param model: (Pipeline)
    The pipeline that made the final model

    :return: (.png images)
    The returned images are being saved in the
    "customer_churn/images/test_results" folder
    '''
    try:
        importances = model.steps[2][1].feature_importances_

        indices = np.argsort(importances)

        fig, axis = plt.subplots(figsize=(8, 20))
        axis.barh(range(len(importances)), importances[indices])
        axis.set_yticks(range(len(importances)))
        _ = axis.set_yticklabels(
            np.array(model[:-1].get_feature_names_out())[indices])
        plt.savefig(RESULT_IMAGES_PATH + 'output_feature_importance.png')
        plt.show()
        logging.info('Execution of feature_importance_plot: SUCCESS')

    except AttributeError:
        logging.error(
            'Exec of feature importance:FAILED.Looks the input you are trying to pass isnt correct')
        return None


if __name__ == "__main__":
    logging.info('About to start the model train step of the system')
    FINAL_MODEL, _ = train_model()
    print('The train_model function has been executed')

    feature_importance_plot(FINAL_MODEL)
    print('The feature_importance_plot function has been executed: Executed System!')
    logging.info('Done executing the model train step: SUCCESS')
