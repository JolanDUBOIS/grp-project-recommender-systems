import pandas as pd

from src.recommender_systems.baseline import BaselineMostClicked
from src.recommender_systems.collaborative_filtering.als_matrix_fact import ALSMatrixFactorization
from src.recommender_systems.collaborative_filtering.item_item import ItemItemCollaborativeFiltering
from src.recommender_systems.hybrid.hybrid_item_item import HybridItemItemCollabFiltering
from src.recommender_systems.feature_based.content_similarity import ContentBasedFiltering
from src.data_normalization import data_normalization
import src.evaluation.helper_functions as helper

K = 10

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
        model = HybridItemItemCollabFiltering()
    elif model_type == "content_based":
        model = ContentBasedFiltering()
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
        recall_sum = 0
        mrr_sum = 0
        countable_users = 0

        for user_id in user_ids:
            prediction = model.predict(user_id, data['behaviors']['Time'].iloc[0], K)

            actual = (
                processed_validation_data.loc[processed_validation_data["User ID"] == user_id, "News IDs"].values[0]
                if user_id in processed_validation_data["User ID"].values
                else []
            )

            if len(actual) != 0:
                countable_users += 1
                precision_sum += helper.precision_at_k(prediction, actual, k=K)
                recall_sum  += helper.recall_at_k(prediction, actual, k=K)
                mrr_sum += helper.mrr_at_k(prediction, actual, k=K)

        average_precision = precision_sum / countable_users
        print(f"Average Precision@{K}: {average_precision:.4f}")

        average_recall = recall_sum / countable_users
        print(f"Average Recall@{K}: {average_recall:.4f}")

        average_mrr = mrr_sum / countable_users
        print(f"Average MRR@{K}: {average_mrr:.4f}")
