import pandas as pd


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
    # Ensure timestamp column is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])

    # Determine the minimum timestamp to start bucketing
    min_time = df[time_col].min()

    # Compute the bucket start time for each row
    df["time_bucket"] = df[time_col].apply(lambda x: min_time + pd.Timedelta(seconds=((x - min_time).total_seconds() // time_window) * time_window))

    # Sort bucket timestamps to maintain order
    sorted_buckets = sorted(df["time_bucket"].unique())

    # Assign index values (0, 1, 2, ...) to each time bucket
    bucket_index_mapping = {time: index for index, time in enumerate(sorted_buckets)}

    # Map bucket timestamps to indexed values
    df["bucket_index"] = df["time_bucket"].map(bucket_index_mapping)

    # Group by bucket index and store each subset in a dictionary
    indexed_buckets = {index: group.drop(columns=["time_bucket", "bucket_index"]) for index, group in df.groupby("bucket_index")}

    return indexed_buckets

def get_arranged_validation_data(validation_bucket, impressions):
    impression_ids = validation_bucket["Impression ID"]
    impression_data = impressions[impressions["Impression ID"].isin(impression_ids)]

    df_merged = impression_data.merge(validation_bucket[["Impression ID", "User ID", "Time"]], on="Impression ID")
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

    return df_merged

def precision_at_k(predicted, actual, k=10):
    predicted_at_k = predicted[:k]
    relevant = set(predicted_at_k) & set(actual)
    precision = len(relevant) / k
    return precision

def recall_at_k(predicted, actual, k=10):
    if not actual:
        return 0.0  # avoid division by zero
    predicted_at_k = predicted[:k]
    relevant = set(predicted_at_k) & set(actual)
    recall = len(relevant) / len(actual)
    return recall


def mrr_at_k(predicted, actual, k=10):
    predicted_at_k = predicted[:k]
    for rank, item in enumerate(predicted_at_k, start=1):
        if item in actual:
            return 1.0 / rank
    return 0.0

