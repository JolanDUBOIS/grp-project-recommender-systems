import pandas as pd
from tqdm import tqdm

import src.evaluation.helper_functions as helper
from src.data_normalization import data_normalization
from src.recommender_systems import (
    BaselineMostClicked,
    ALSMatrixFactorization,
    ItemItemCollaborativeFiltering,
    TrueHybrid,
    ContentBasedFiltering
)

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
        model = TrueHybrid(alpha=0.5)
    elif model_type == "content_based":
        model = ContentBasedFiltering()
    else:
        model = BaselineMostClicked()

    data_buckets = helper.split_data(data['behaviors'], 'Time', TIME_WINDOW)
    for i in range(3, len(data_buckets) - 1):
        recommended_items = []
        all_items = data['news']['News ID'].values

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

        precision_sum = 0
        recall_sum = 0
        mrr_sum = 0
        acc_sum = 0
        diversity_sum = 0
        countable_users = 0

        for _, row in tqdm(validation_bucket.iterrows(), total=validation_bucket.shape[0], desc="Making predictions:"):
            prediction = model.predict(row["User ID"], row['Time'], K)
            recommended_items.extend(prediction)

            actual = (
                processed_validation_data.loc[processed_validation_data["User ID"] == row["User ID"], "News IDs"].values[0]
                if row["User ID"] in processed_validation_data["User ID"].values
                else []
            )

            if len(actual) != 0:
                countable_users += 1
                precision_sum += helper.precision_at_k(prediction, actual, k=K)
                recall_sum  += helper.recall_at_k(prediction, actual, k=K)
                mrr_sum += helper.mrr_at_k(prediction, actual, k=K)
                acc_sum += helper.accuracy(prediction, actual)
                diversity_sum += helper.calculate_diversity(prediction, data['news'])

        average_precision = precision_sum / countable_users
        print(f"Average Precision@{K}: {average_precision:.4f}")

        average_recall = recall_sum / countable_users
        print(f"Average Recall@{K}: {average_recall:.4f}")

        average_mrr = mrr_sum / countable_users
        print(f"Average MRR@{K}: {average_mrr:.4f}")

        average_acc = acc_sum / countable_users
        print(f"Average Accuracy: {average_acc:.4f}")

        diversity_acc = diversity_sum / countable_users
        print(f"Average Diversity@{K}: {diversity_acc:.4f}")

        recommended_items = set(recommended_items)
        coverage = helper.coverage(recommended_items, all_items)
        print(f"Coverage: {coverage:.4f}")

        diversity = helper.calculate_diversity(recommended_items, data["news"])
        print(f"Average Diversity: {diversity:.4f}")



def validation_set_workflow(model_type="baseline"):
    if model_type == "baseline":
        model = BaselineMostClicked()
    elif model_type == "als":
        model = ALSMatrixFactorization()
    elif model_type == "itemitem":
        model = ItemItemCollaborativeFiltering()
    elif model_type == "hybrid":
        model = TrueHybrid(alpha=0.6)
    elif model_type == "content_based":
        model = ContentBasedFiltering()
    else:
        model = BaselineMostClicked()

    data, embeddings = data_normalization(validation=False, try_load=True)
    validation_data, _ = data_normalization(validation=True, try_load=False)

    model.fit(data, embeddings)

    precision_sum = 0
    recall_sum = 0
    mrr_sum = 0
    acc_sum = 0
    diversity_sum = 0
    countable_users = 0
    recommended_items = []

    all_items = data['news']['News ID'].values

    validation_behavior = validation_data["behaviors_val"]
    processed_validation_data = helper.get_arranged_validation_data(validation_behavior, validation_data["impressions_val"])

    for _, row in tqdm(validation_behavior.iterrows(), total=validation_behavior.shape[0], desc="Making predictions:"):
        prediction = model.predict(row["User ID"], pd.to_datetime(row['Time']), K)
        recommended_items.extend(prediction)

        actual = (
            processed_validation_data.loc[processed_validation_data["User ID"] == row["User ID"], "News IDs"].values[0]
            if row["User ID"] in processed_validation_data["User ID"].values
            else []
        )

        if len(actual) != 0:
            countable_users += 1
            precision_sum += helper.precision_at_k(prediction, actual, k=K)
            recall_sum += helper.recall_at_k(prediction, actual, k=K)
            mrr_sum += helper.mrr_at_k(prediction, actual, k=K)
            acc_sum += helper.accuracy(prediction, actual)
            diversity_sum += helper.calculate_diversity(prediction, data['news'])

    average_precision = precision_sum / countable_users
    print(f"Average Precision@{K}: {average_precision:.4f}")

    average_recall = recall_sum / countable_users
    print(f"Average Recall@{K}: {average_recall:.4f}")

    average_mrr = mrr_sum / countable_users
    print(f"Average MRR@{K}: {average_mrr:.4f}")

    average_acc = acc_sum / countable_users
    print(f"Average Accuracy: {average_acc:.4f}")

    diversity_acc = diversity_sum / countable_users
    print(f"Average Diversity@{K}: {diversity_acc:.4f}")

    recommended_items = set(recommended_items)
    coverage = helper.coverage(recommended_items, all_items)
    print(f"Coverage: {coverage:.4f}")

    diversity = helper.calculate_diversity(recommended_items, data["news"])
    print(f"Average Diversity: {diversity:.4f}")
