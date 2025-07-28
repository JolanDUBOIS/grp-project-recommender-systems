import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from . import logger


def split_data(df, time_col, time_window):
    """
    Splits a DataFrame into indexed time-window-based buckets using a timestamp column.

    Parameters:
        df (pd.DataFrame): The dataset containing a timestamp column.
        time_col (str): Name of the timestamp column.
        time_window (int): The time window in seconds for bucketing.

    Returns:
        dict: A dictionary where keys are integers (0,1,2...) representing bucket indices,
              and values are DataFrames for each bucket.
    """
    try:
        # Ensure timestamp column is in datetime format
        df[time_col] = pd.to_datetime(df[time_col])
        logger.debug(f"Converted {time_col} to datetime format.")
    except Exception as e:
        logger.error(f"Error converting {time_col} to datetime: {e}")
        raise

    # Determine the minimum timestamp to start bucketing
    min_time = df[time_col].min()
    logger.debug(f"Minimum timestamp: {min_time}, Time window: {time_window} seconds.")

    # Compute the bucket start time for each row
    df["time_bucket"] = df[time_col].apply(lambda x: min_time + pd.Timedelta(seconds=((x - min_time).total_seconds() // time_window) * time_window))

    # Sort bucket timestamps to maintain order
    sorted_buckets = sorted(df["time_bucket"].unique())
    logger.debug(f"Generated {len(sorted_buckets)} time buckets.")

    # Assign index values (0, 1, 2, ...) to each time bucket
    bucket_index_mapping = {time: index for index, time in enumerate(sorted_buckets)}

    # Map bucket timestamps to indexed values
    df["bucket_index"] = df["time_bucket"].map(bucket_index_mapping)

    # Group by bucket index and store each subset in a dictionary
    indexed_buckets = {index: group.drop(columns=["time_bucket", "bucket_index"]) for index, group in df.groupby("bucket_index")}
    logger.debug(f"Split data into {len(indexed_buckets)} indexed buckets.")

    return indexed_buckets

def get_arranged_validation_data(validation_bucket, impressions):
    impression_ids = validation_bucket["Impression ID"]
    try:
        impression_data = impressions[impressions["Impression ID"].isin(impression_ids)]
        logger.debug(f"Filtered impressions to {len(impression_data)} rows.")
    except Exception as e:
        logger.error(f"Error filtering impressions: {e}")
        raise

    df_merged = impression_data.merge(validation_bucket[["Impression ID", "User ID", "Time"]], on="Impression ID")
    logger.debug(f"Merged validation bucket with impressions. Resulting rows: {len(df_merged)}.")
    df_merged["Clicked"] = pd.to_numeric(df_merged["Clicked"], errors='coerce')
    df_merged = df_merged[df_merged["Clicked"] == 1]

    df_merged.drop(columns=["Impression ID", "Clicked"], inplace=True)
    df_merged = df_merged[["User ID", "News ID", "Time"]]

    df_merged = df_merged.sort_values(by="User ID")
    df_merged = df_merged.reset_index(drop=True)

    df_merged = (
        df_merged
        .groupby("User ID")["News ID"]
        .agg(list)
        .reset_index()
        .rename(columns={"News ID": "News IDs"})
    )
    logger.debug(f"Arranged validation data for {df_merged['User ID'].nunique()} unique users.")

    return df_merged

def precision_at_k(predicted, actual, k=10):
    predicted_at_k = predicted[:k]
    relevant = set(predicted_at_k) & set(actual)
    precision = len(relevant) / k
    logger.debug(f"Calculating precision at k={k}. Predicted: {predicted[:k]}, Actual: {actual}.")
    return precision

def recall_at_k(predicted, actual, k=10):
    if not actual:
        return 0.0  # avoid division by zero
    predicted_at_k = predicted[:k]
    relevant = set(predicted_at_k) & set(actual)
    recall = len(relevant) / len(actual)
    logger.debug(f"Calculating recall at k={k}. Predicted: {predicted[:k]}, Actual: {actual}.")
    return recall


def mrr_at_k(predicted, actual, k=10):
    predicted_at_k = predicted[:k]
    for rank, item in enumerate(predicted_at_k, start=1):
        if item in actual:
            logger.debug(f"Calculating MRR at k={k}. Predicted: {predicted[:k]}, Actual: {actual}.")
            return 1.0 / rank
    return 0.0

def accuracy(y_true, y_pred):
    """
    Computes accuracy as the proportion of correct predictions.
    Assumes y_true and y_pred are lists or arrays of equal length.
    """
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    logger.debug(f"Calculating accuracy. Total samples: {len(y_true)}.")
    return correct / len(y_true) if y_true else 0


def coverage(recommended_items, total_items):
    logger.debug(f"Calculating coverage. Recommended items: {len(recommended_items)}, Total items: {len(total_items)}.")
    return len(recommended_items) / len(total_items)

def calculate_diversity(recommended_items, news_df):
    logger.debug(f"Calculating diversity for {len(recommended_items)} recommended items.")
    news_df = news_df.copy()
    news_df = news_df[news_df["News ID"].isin(recommended_items)]
    news_df["content"] = news_df["Title"] + " " + news_df["Abstract"]
    news_df["content"] = news_df["content"].fillna("")
    vectorizer = TfidfVectorizer(max_features=20)
    if not vectorizer:
        return 0

    news_df = news_df.dropna(subset=["content"])  # drop rows where content is NaN
    news_df["content"] = news_df["content"].astype(str)  # force everything to string

    if news_df["content"].empty:
        logger.error("No content available for diversity calculation.")
        return 0.0
    else:
        X = vectorizer.fit_transform(news_df["content"])

    try:
        article_embeddings = X.toarray()
    except AttributeError:
        return 0.0

    sim_matrix = cosine_similarity(article_embeddings)

    n = len(recommended_items)
    sim_sum = 0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            sim_sum += sim_matrix[i][j]
            pair_count += 1

    avg_similarity = sim_sum / pair_count if pair_count > 0 else 0
    diversity = 1 - avg_similarity
    logger.debug(f"Average similarity: {avg_similarity}, Diversity: {diversity}.")

    return diversity

