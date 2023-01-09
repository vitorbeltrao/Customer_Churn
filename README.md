# Predict Customer Churn

## Table of Contents

1. [Project Description](#Description)
2. [Files Description](#files)
3. [Running Files](#running)
4. [Licensing and Authors](#licensingandauthors)
***

## Project Description <a name="Description"></a>

Assuming that the objective of any business is to increase the number of customers, it is very important to understand that for this to happen, it is not enough just to attract new customers, it is necessary to keep the ones that are already there.

Keeping customers can be easier than attracting new ones, as they may be less interested in the services offered and do not have a previous relationship with the company. Furthermore, keeping a customer costs the company less than attracting new ones.

By predicting the turnover of existing customers, that is, by carrying out churn prediction, it is possible to anticipate when a customer intends to cancel a service and thus react in time to keep it (creating special offers, for example).

**The challenge is to predict and take measures to reduce churn as much as possible, thus ensuring satisfied customers who do not intend to stop subscribing to the product.**
***

## Files Description <a name="files"></a>

In "customer_churn" repository we have:

* **customer_churn_model folder**: Inside this folder, we have all the files needed to run the entire model pipeline, from raw data collection to final predictions for never-before-seen data, as well as the log files, which are used to monitor how the model is running. These are the final files for the production environment.

* **data folder**: This is the folder that has the raw dataset collected to be used as input for creating the entire machine learning system.

* **images folder**: This is the folder where we keep the images of some important results generated during the construction of the system. It contains images of important graphics used in the exploratory analysis stage and in the analysis stage of the final metrics of the model.

* **tests folder**: This is the folder where the tests used to test all the functions we created to run the system from end to end are stored. The tests were created with the help of the *pytest* library.

* **notebooks (.ipynb files)**: The four .ipynb files present were used to create the entire prototype of the model. They basically work as an experimentation environment so that we can later create the production environment.

* **requirements.txt**: File with the list of all dependencies used to create the system.
***

## Running Files <a name="running"></a>

### Get the data

As a good practice, the .csv file is not included in the package.
- Save the `bank_data.csv` file in the `customer_churn/data` directory
- Make sure the directory and file's name is correct

### Run 

- Add `customer_churn` *and* `customer_churn_model` paths to your system PYTHONPATH
- Install dependencies through requirements:

    `pip install -r requirements.txt`

### Train model

- To train the model, run in order:

    `python customer_churn_model/etl.py`

    `python customer_churn_model/train.py`

    The trained model will be saved in `customer_churn_model/customer_churn_model.pkl` and will be ready to be used to make predictions. In addition, an image with the importance of the variables used to train the model will also be saved.

### Test model

- To test the trained model, the model needs to be trained and saved (steps above)
- To test the model on the test dataset and verify it based on key evaluation metrics, run:

    `python customer_churn_model/test.py`

### Make predictions on data never seen before

- To make predictions, the model needs to be trained and saved (steps above)
- Add a new .csv file as `new_data.csv` to `customer_churn_model/new_data` to make predictions on (this file should have the same structure to the dataset where the model was trained on, without *"Attrition_Flag"* column)
- then run:

    `python customer_churn_model/predict.py`

### Testing

- Run the tests

    `pytest`
***

## Licensing and Author <a name="licensingandauthors"></a>

Vítor Beltrão - Data Scientist

Reach me at: 

- vitorbeltraoo@hotmail.com

- [linkedin](https://www.linkedin.com/in/v%C3%ADtor-beltr%C3%A3o-56a912178/)

- [github](https://github.com/vitorbeltrao)

- [medium](https://pandascouple.medium.com)

Licensing: [GNU GENERAL PUBLIC LICENSE](https://github.com/vitorbeltrao/customer_churn/blob/main/LICENSE)