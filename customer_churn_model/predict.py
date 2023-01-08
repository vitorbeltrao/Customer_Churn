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

# import the pkl file
_churn_pipeline = joblib.load(SAVE_PKL_FILE_NAME)


def predict_churn(input_data: str) -> dict:
    '''Function to make predictions on new data. To use this function on
    new data, you must upload your data in the "customer_churn_model/new_data"
    folder and your file must be called "new_data"

    :param input_data: (str)
    The path of the file that should call "new_data"

    :return: (dict)
    The prediction of your data
    '''
    try:
        results = {'predictions': None}
        predictions = _churn_pipeline.predict(import_data(input_data))
        results = {'predictions': predictions}

        print('The results are:', results)
        logging.info('Execution of predict_churn: SUCCESS')
        return results

    except FileNotFoundError:
        logging.error("Execution of predict_churn: The file wasn't found")
        return None

    except TypeError:
        logging.error(
            'Execution of predict_churn: Your input is not correctly')
        return None


if __name__ == "__main__":
    logging.info('About to start the prediction')
    print('predicting on new data...')

    predict_churn(NEW_DATA)

    print('The predict_churn function has been executed: Executed System!')
    logging.info('Done executing the predictions: SUCCESS')
