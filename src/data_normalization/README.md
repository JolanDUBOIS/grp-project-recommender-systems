# Data Normalization

This part of the data pipeline enables raw data to be normalized in an easy-to-use relational database.

## Overview of the Code

- The `data_normalization` function processes raw data from the MIND dataset. It loads the raw data, normalizes it (splits history and impressions, normalizes news titles and abstracts), and then optionally saves the processed data for future use.
- The dataset includes two main parts: news data and behavior data (user interactions with news items).
- Embedding files are also read into dictionaries to store entity and relation embeddings.
- The processed data can be saved in a folder called `data_normalized` for reuse, avoiding the need to reprocess the raw data.

## Command-Line Inferface (CLI)

To run the script for data normalization, you need to use the command-line interface. The main entry point is `__main__.py`, which handles the arguments and triggers the normalization process. You can use one of the following commands:

```bash
python -m src -dn
python -m src --data-normalization
```

**Optional arguments:**
- `--validation` or `-val`: if specified, the script will normalize the validation data instead of the training data.

## Key Steps in the Normalization Process:

1. Read Raw Data: The raw data (news, behaviors, and embeddings) is read from the MIND dataset.

2. Normalize Behaviors: The "History" and "Impressions" columns are expanded to split space-separated values into lists and rows.

3. Normalize News: The "Title Entities" and "Abstract Entities" columns are parsed, expanding entity lists into rows and normalizing nested entities.

4. Save Processed Data: The processed data is saved as CSV files in the data_normalized folder if the save flag is enabled.

## Additional Information

**Flexibility**: The normalization process can be skipped by setting `try_load=True` and providing pre-processed data in the data_normalized folder.

**Environment Variables**: The script expects an environment variable `MIND_DATA_FOLDER_PATH` to specify the location of the MIND dataset on your system. To set the environment variable, create a `.env` file and write:

```bash
export MIND_DATA_FOLDER_PATH="path/to/the/folder"
```

Then, execute the command `source .env`.