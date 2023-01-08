'''
Author: Vitor Abdo

This .py file is for testing using the best model trained
'''
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from decouple import config
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, roc_auc_score
from etl import import_data


# config
TARGET_AFTER_ETL = config('TARGET_AFTER_ETL')
TEST_DATASET = config('TEST_DATASET')
SAVE_PKL_FILE_NAME = config('SAVE_PKL_FILE_NAME')
RESULT_IMAGES_PATH = config('RESULT_IMAGES_PATH')

# import the pkl file
_churn_pipeline = joblib.load(SAVE_PKL_FILE_NAME)


def test_model():
    '''Function that tests the model with the test data and shows
    the results according to various metrics

    :return: (.png images)
    The returned images are being saved in the
    "customer_churn/images/test_results" folder
    '''
    try:
        # read testing data
        df_test_transformed = import_data(TEST_DATASET)

        # select only the features that we are going to use and make
        # predictions
        X_test = df_test_transformed.drop([TARGET_AFTER_ETL], axis=1)
        y_test = df_test_transformed[TARGET_AFTER_ETL]
        final_predictions = _churn_pipeline.predict(X_test)

        # print classification report
        print(classification_report(y_test, final_predictions))

        # plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(
                y_test,
                final_predictions,
                normalize='true'),
            annot=True,
            ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predict")
        plt.savefig(RESULT_IMAGES_PATH + 'output_confusionmatrix.png')
        plt.show()

        # print and plot roc_auc
        print("AUC: {:.4f}\n".format(roc_auc_score(y_test, final_predictions)))

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        lgbm_disp = RocCurveDisplay.from_estimator(
            _churn_pipeline, X_test, y_test, ax=ax, alpha=0.8)
        lgbm_disp.plot(ax=ax, alpha=0.8)
        plt.savefig(RESULT_IMAGES_PATH + 'output_roccurve.png')
        plt.show()

        logging.info("Execution of test model: SUCCESS")

    except FileNotFoundError:
        logging.error(
            "No such file or directory. Need to create the folder to save the images")
        return None


if __name__ == "__main__":
    logging.info('About to start the model test step of the system')
    test_model()
    print('The test_model function has been executed: Executed System!')
    logging.info('Done executing the model test step: SUCCESS')
