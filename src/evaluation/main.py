import pandas as pd

from src.recommender_systems.baseline import BaselineMostClicked
from src.recommender_systems.collaborative_filtering.als_matrix_fact import ALSMatrixFactorization
from src.recommender_systems.collaborative_filtering.item_item import ItemItemCollaborativeFiltering
from src.data_normalization import data_normalization
import src.evaluation.helper_functions as helper

K = 10

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
def sliding_window_workflow(data, embeddings, model_type="baseline", TIME_WINDOW=86400):
    ### a sliding window approach at training and testing
    ### meant to work for all models

    if model_type == "baseline":
        model = BaselineMostClicked()
    elif model_type == "als":
        model = ALSMatrixFactorization()
    elif model_type == "itemitem":
        model = ItemItemCollaborativeFiltering()
    elif model_type == "hybrid":
        raise ("this will be used for hybrid approach - not implemented yet!")
    else:
        model = BaselineMostClicked()

    data_buckets = helper.split_data(data['behaviors'], 'Time', TIME_WINDOW)
    for i in range(len(data_buckets) - 1):
        training_bucket = data_buckets[i]
        validation_bucket = data_buckets[i + 1]
        processed_validation_data = helper.get_arranged_validation_data(validation_bucket, data["impressions"])

        training_data = {
            'behaviors': training_bucket,
            'impressions': data['impressions'],
            'news': data['news'],
            'history': data['history'],
            'title_entities': data['title_entities'],
            'abstract_entities': data['abstract_entities'],
        }

        model.fit(training_data, embeddings)
        user_ids = training_bucket["User ID"].tolist()

        precision_sum = 0

        for user_id in user_ids:
            prediction = model.predict(user_id, data['behaviors']['Time'].iloc[0], K)

            actual = (
                processed_validation_data.loc[processed_validation_data["User ID"] == user_id, "News IDs"].values[0]
                if user_id in processed_validation_data["User ID"].values
                else []
            )

            precision_sum += helper.precision_at_k(prediction, actual, k=K)

        average_precision = precision_sum / len(user_ids)
        print(f"Average Precision@{K}: {average_precision:.4f}")
