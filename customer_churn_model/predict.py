'''
Author: Vitor Abdo

This .py file is for making predictions for new data
'''
import logging
import joblib
from decouple import config
from etl import import_data

# config
SAVE_PKL_FILE_NAME = config('SAVE_PKL_FILE_NAME')
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')
NEW_DATA = config('NEW_DATA')
NEW_TEST_DATA = config('NEW_TEST_DATA')

# import the pkl file
_churn_pipeline = joblib.load(SAVE_PKL_FILE_NAME)


def predict_churn(input_data: str) -> dict:
    '''
    '''
    raw_df = import_data(input_data)
    results = {'predictions': None}

    predictions = _churn_pipeline.predict(X=raw_df)

    results = {
        'predictions': predictions,
    }
    print('Execution of predict_churn: SUCCESS', results)
    return results


if __name__ == "__main__":
    logging.info('About to start the prediction')

    try:
        raw_df = import_data(NEW_DATA)
        print('predicting on new data...')

        X_test = raw_df.copy()
        predict_churn(X_test)

        logging.info('Done executing the predictions: SUCCESS')

    except FileNotFoundError:
        print("Execution of predict_churn: The file wasn't found")
        logging.error("Execution of predict_churn: The file wasn't found")

    except TypeError:
        print('Execution of predict_churn: Your input is not correctly')
        logging.error('Execution of predict_churn: Your input is not correctly')






    