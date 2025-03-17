import os, ast
from pathlib import Path

import pandas as pd


SMALL_MIND_SETS = {
    "train": "MINDsmall_train",
    "validation": "MINDsmall_dev",
}

MIND_DATA_FILES = {
    "news": {"filename": "news.tsv", "columns": ["News ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title Entities", "Abstract Entities"]},
    "behaviors": {"filename": "behaviors.tsv", "columns": ["Impression ID", "User ID", "Time", "History", "Impressions"]},
}
MIND_EMBEDDING_FILES = {
    "entity_embedding": {"filename": "entity_embedding.vec"},
    "relation_embedding": {"filename": "relation_embedding.vec"}
}

def get_data_folder_path() -> Path:
    """
    Retrieves the folder path where MIND dataset is stored from the environment variable.

    This function attempts to retrieve the data folder path from the environment variable 
    `MIND_DATA_FOLDER_PATH`, and raises an error if it's not set or if the folder doesn't exist.

    Returns:
        Path: The path to the data folder.

    Raises:
        TypeError: If the `MIND_DATA_FOLDER_PATH` environment variable is not set.
        FileNotFoundError: If the folder does not exist at the specified path.
    """
    try:
        data_folder_path = Path(os.getenv("MIND_DATA_FOLDER_PATH"))
    except TypeError:
        raise TypeError("DATA_FOLDER_PATH environment variable is not set.")

    if not data_folder_path.exists():
        raise FileNotFoundError(f"Data folder not found at {data_folder_path}")

    return data_folder_path

def read_tsv(file_path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Reads a tab-separated values (TSV) file and returns it as a pandas DataFrame.

    Args:
        file_path (Path): Path to the TSV file.
        columns (list[str]): List of column names to be used for the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the TSV file.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
    """

    df = pd.read_csv(file_path, sep="\t", header=None, names=columns)
    return df

def read_embedding(file_path: Path) -> dict:
    """
    Reads an embedding file and returns the embeddings as a dictionary.

    Args:
        file_path (Path): Path to the embedding file.

    Returns:
        dict: A dictionary where keys are the entity IDs and values are the embedding vectors (lists of floats).

    Raises:
        FileNotFoundError: If the embedding file does not exist at the specified path.
    """
    with open(file_path) as f:
        lines = f.readlines()
    embedding = {}
    for line in lines:
        line = line.strip().split()
        embedding[line[0]] = list(map(float, line[1:]))
    return embedding

def read_raw_data(data_folder_path: Path, only_embeddings: bool=False, suffix: str='') -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """
    Reads and returns the raw data from the MIND dataset (news, behaviors, and embeddings).

    Args:
        data_folder_path (Path): The path to the folder containing the dataset.
        only_embeddings (bool): If True, only reads the embeddings and not the data. Defaults to False.
        suffix (str): A suffix to append to the keys of the data and embeddings dictionaries. Defaults to an empty string.

    Returns:
        tuple: A tuple containing two dictionaries:
            - A dictionary of DataFrames with keys for 'news' and 'behaviors'.
            - A dictionary of embeddings with keys for 'entity_embedding' and 'relation_embedding'.

    Raises:
        FileNotFoundError: If any of the expected files are missing in the dataset folder.
    """
    data = {}
    embeddings = {}

    if not only_embeddings:
        for file_key, file_info in MIND_DATA_FILES.items():
            file_path = data_folder_path / file_info["filename"]
            if not file_path.exists():
                raise FileNotFoundError(f"File not found at {file_path}")
            data[f"{file_key}{suffix}"] = read_tsv(file_path, file_info["columns"])

    for file_key, file_info in MIND_EMBEDDING_FILES.items():
        file_path = data_folder_path / file_info["filename"]
        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")
        embeddings[f"{file_key}{suffix}"] = read_embedding(file_path)

    return data, embeddings

def normalize_behavior(data: dict[str, pd.DataFrame], suffix: str='') -> dict[str, pd.DataFrame]:
    """
    Normalize the 'History' and 'Impressions' columns in the behaviors DataFrame.

    This function:
    - Splits space-separated 'History' and 'Impressions' strings into lists.
    - Expands these lists into separate rows (one entry per row).
    - Splits impression records into 'News ID' and a binary 'Clicked' flag.

    Args:
        data (dict[str, pd.DataFrame]): Dictionary containing a 'behaviors' DataFrame with history and impression data.

    Returns:
        dict[str, pd.DataFrame]: The updated data dictionary including the expanded history and impressions tables.
    """
    behaviors_df = data[f"behaviors{suffix}"]

    # Convert space-separated strings into lists
    behaviors_df['History'] = behaviors_df['History'].str.split()
    behaviors_df['Impressions'] = behaviors_df['Impressions'].str.split()

    # Expand lists into multiple rows
    history_df = behaviors_df[['Impression ID', 'History']].explode('History')
    impressions_df = behaviors_df[['Impression ID', 'Impressions']].explode('Impressions')

    # Split each impression entry (e.g., "news_id-clicked") into two separate columns
    impressions_df['Impressions'] = impressions_df['Impressions'].str.split('-')
    impressions_df[['News ID', 'Clicked']] = pd.DataFrame(impressions_df['Impressions'].tolist(), index=impressions_df.index)
    impressions_df.drop(columns=['Impressions'], inplace=True)

    return {**data, f"history{suffix}": history_df, f"impressions{suffix}": impressions_df}

def normalize_news(data: dict[str, pd.DataFrame], suffix: str='') -> dict[str, pd.DataFrame]:
    """
    Normalize the 'Title Entities' and 'Abstract Entities' columns in the news DataFrame.

    This function:
    - Parses string representations of lists in 'Title Entities' and 'Abstract Entities' into actual lists.
    - Expands entity lists into separate rows (one entity per row).
    - Normalizes nested dictionaries within each entity into separate columns.

    Args:
        data (dict[str, pd.DataFrame]): Dictionary containing a 'news' DataFrame with entity columns.

    Returns:
        dict[str, pd.DataFrame]: The updated data dictionary including the normalized entity tables.
    """
    news_df = data[f"news{suffix}"]

    # Convert string representations of lists into actual lists and replace NaN values with empty lists
    news_df['Title Entities'] = news_df['Title Entities'].where(news_df['Title Entities'].notna(), '[]').apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    news_df['Abstract Entities'] = news_df['Abstract Entities'].where(news_df['Abstract Entities'].notna(), '[]').apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Expand entity lists into multiple rows
    title_entities_df = news_df[['News ID', 'Title Entities']].explode('Title Entities')
    abstract_entities_df = news_df[['News ID', 'Abstract Entities']].explode('Abstract Entities')

    # Flatten entity dictionaries into separate columns
    title_entities_df = title_entities_df.join(pd.json_normalize(title_entities_df.pop('Title Entities')))
    abstract_entities_df = abstract_entities_df.join(pd.json_normalize(abstract_entities_df.pop('Abstract Entities')))

    return {**data, f"title_entities{suffix}": title_entities_df, f"abstract_entities{suffix}": abstract_entities_df}

def data_normalization(validation: bool=False, try_load: bool=True, save: bool=False) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """
    Normalizes and loads or saves the data depending on whether pre-normalized files exist.

    - If the 'data_normalized' subfolder exists and `try_load` is True, it loads the normalized data from CSV files.
    - If the subfolder doesn't exist, or if `try_load` is False, it normalizes the raw data and optionally saves it to the subfolder.

    Args:
        validation (bool): TODO - by default we read and nomalized the training set
        try_load (bool): If True, attempts to load pre-normalized data from the 'data_normalized' folder. 
                          If False, the data will be normalized regardless of the subfolder's existence.
        save (bool): If True, saves the normalized data into the 'data_normalized' folder.

    Returns:
        TODO
    """
    data_folder_path = get_data_folder_path()
    save_subfolder = data_folder_path / 'data_normalized'
    mind_subfolder = SMALL_MIND_SETS["validation"] if validation else SMALL_MIND_SETS["train"]
    read_subfolder = data_folder_path / mind_subfolder
    suffix = "_val" if validation else ""

    if try_load and save_subfolder.exists() and save_subfolder.is_dir():
        data = {}
        for file in save_subfolder.iterdir():
            if file.is_file() and file.suffix == ".csv":
                data[file.stem] = pd.read_csv(file)
        _, embeddings = read_raw_data(read_subfolder, only_embeddings=True, suffix=suffix)

    else:
        data, embeddings = read_raw_data(read_subfolder, suffix=suffix)
        data = normalize_behavior(data, suffix=suffix)
        data = normalize_news(data, suffix=suffix)

        if save:
            save_subfolder.mkdir(parents=True, exist_ok=True)
            for key, df in data.items():
                df.to_csv(save_subfolder / f"{key}.csv", index=False)

    return data, embeddings
