import pandas as pd

from src.recommender_systems.baseline import BaselineMostClicked
from src.recommender_systems.collaborative_filtering import ALSMatrixFactorization
from src.data_normalization import data_normalization
import src.evaluation.helper_functions as helper


def compare_baseline_als():
    """ Compare the baseline and ALS matrix factorization models. """
    # Load the data
    data, _ = data_normalization(validation=False, try_load=True)

    # Initialize the models
    baseline = BaselineMostClicked()
    als = ALSMatrixFactorization()

    # Split the data
    data['behaviors'] = data['behaviors'].sort_values('Time')
    split_index = int(0.8 * len(data['behaviors']))
    data_train = {
        'behaviors': data['behaviors'].iloc[:split_index],
        'Impressions': data['Impressions'],
        'news': data['news'],
        'History': data['History'],
        'Title entities': data['Title entities'],
        'Abstract entities': data['Abstract entities'],
    }
    data_test = {
        'behaviors': data['behaviors'].iloc[split_index:],
        'Impressions': data['Impressions'],
        'news': data['news'],
        'History': data['History'],
        'Title entities': data['Title entities'],
        'Abstract entities': data['Abstract entities'],
    }

    # Fit the models
    baseline.fit(data_train)
    als.fit(data_train)

    # Predict
    user_id = 'U13740'
    time = data_test['behaviors']['Time'].iloc[0]
    k = 10
    baseline_recommendations = baseline.predict(user_id, time, k)
    als_recommendations = als.predict(user_id, time, k)

    # Compare

### takes model_type as an argument - that is the model that is being trained and tested
### default TIME_WINDOW - 1 day (in seconds)
def sliding_window_workflow(model_type="baseline", TIME_WINDOW=86400):
    ### a sliding window approach at training and testing
    ### meant to work for all models

    data, _ = data_normalization(validation=False, try_load=True)

    if model_type == "baseline":
        model = BaselineMostClicked()
    elif model_type == "als":
        model = ALSMatrixFactorization()
    elif model_type == "hybrid":
        raise ("this will be used for hybrid approach - not implemented yet!")
    else:
        model = BaselineMostClicked()

    data_buckets = helper.split_data(data['behaviors'], 'Time', TIME_WINDOW)
    for i in range(len(data_buckets) - 1):
        training_bucket = data_buckets[i]
        validation_bucket = data_buckets[i + 1]

        training_data = {
            'behaviors': training_bucket,
            'Impressions': data['Impressions'],
            'news': data['news'],
            'History': data['History'],
            'Title entities': data['Title entities'],
            'Abstract entities': data['Abstract entities'],
        }

        model.fit(training_data)


    # Predict
    user_id = 'U13740'
    time = data_test['behaviors']['Time'].iloc[0]
    k = 10
    baseline_recommendations = baseline.predict(user_id, time, k)
    als_recommendations = als.predict(user_id, time, k)

    # Compare
    