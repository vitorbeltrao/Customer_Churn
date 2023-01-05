'''
'''

from customer_churn_model.etl_workflow.etl import import_data
import logging
import pytest


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# config
DATASET = 'data/bank_data.csv'
# EMPTY_DATASET = 'data/Livro.csv'

def test_import_data():
    '''
    '''
    try:
        import_data(DATASET)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError:
        logging.error("Testing import_data: The file wasn't found")

    try:
        assert import_data(DATASET).shape[0] > 0
        assert import_data(DATASET).shape[1] > 0
    except AssertionError:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")


if __name__ == "__main__":
    logging.info('About to start the tests')
    test_import_data()
    logging.info('Done executing the tests')