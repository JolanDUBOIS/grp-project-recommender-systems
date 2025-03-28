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
    df[time_col] = df.sort_values(time_col)
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

def get_arranged_validation_data(validation_bucket):
    behaviour = validation_bucket['behaviors']
    impressions = validation_bucket['Impressions']

    impression_ids = validation_bucket["Impression ID"]
    impression_data = impressions[impressions["Impression ID"].isin(impression_ids)]

    df_merged = impression_data.merge(behaviour[["Impression ID", "User ID", "Time"]], on="Impression ID")

    ## to be completed


def calculate_accuracy(df_true, df_pred):
    """
    Computes accuracy between two DataFrames with User IDs as rows and News IDs as columns.

    Parameters:
        df_true (pd.DataFrame): Ground truth DataFrame (actual clicks).
        df_pred (pd.DataFrame): Predicted DataFrame (recommended clicks).

    Returns:
        float: Accuracy value (0 to 1).
    """
    # Ensure both DataFrames have the same shape
    if df_true.shape != df_pred.shape:
        raise ValueError("DataFrames must have the same shape to compute accuracy")

    # Exclude the "User ID" column for element-wise comparison
    matches = (df_true.iloc[:, 1:] == df_pred.iloc[:, 1:]).sum().sum()
    total_elements = df_true.iloc[:, 1:].size  # Total predictions made

    # Compute accuracy
    accuracy = matches / total_elements
    return accuracy