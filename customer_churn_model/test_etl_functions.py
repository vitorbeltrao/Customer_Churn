'''
'''

from etl_workflow.etl import import_data, transform_data, split_dataset
import logging


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# config
DATASET = 'C:/Users/4YouSee/Desktop/personal_work/customer_churn/data/bank_data.csv'

def test_import_data():
    '''
    '''
    try:
        raw_df = import_data(DATASET)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError:
        logging.error("Testing import_data: The file wasn't found")

    try:
        assert raw_df.shape[0] > 0
        assert raw_df.shape[1] > 0
    except AssertionError:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")


def test_transform_data():
    '''
    '''
    try:
        raw_df = import_data(DATASET)
        df_transformed = transform_data(raw_df)
        logging.info("Testing transform_data: SUCCESS")
    except KeyError:
        logging.error("Testing transform_data: Any variable subject to the transformations of this function was not found")

    try:
        assert df_transformed['Churn'].dtypes == 'int64'
        assert 'Unnamed: 0' not in df_transformed.columns
        assert 'CLIENTNUM' not in df_transformed.columns
        assert 'Attrition_Flag' not in df_transformed.columns
    except AssertionError:
        logging.error("Testing transform_data: The churn column didn't turn into 'int64' or the other columns weren't deleted correctly")


def test_split_dataset():
    '''
    '''
    try:
        raw_df = import_data(DATASET)
        df_transformed = transform_data(raw_df)
        train_set, test_set = split_dataset(df_transformed)
        logging.info("Testing split_dataset: SUCCESS")
    except OSError:
        logging.error("Testing split_dataset: Cannot save file into a non-existent directory")
       
    try:
        assert len(df_transformed) == len(train_set) + len(test_set)
    except AssertionError:
        logging.error("Testing split_dataset: The dataset was not split correctly")
    

if __name__ == "__main__":
    logging.info('About to start the tests')
    test_import_data()
    test_transform_data()
    test_split_dataset()
    logging.info('Done executing the tests')