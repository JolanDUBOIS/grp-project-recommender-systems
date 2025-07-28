import pandas as pd

from . import logger
from src.recommender_systems import RecommenderSystem


class BaselineMostClicked(RecommenderSystem):
    """ Baseline recommender system. """
    # TODO - Rename most popular (cf lecture 06/02)
    # This function is a most popular with a time window which looks a bit like ring buffer but not really
    
    def __init__(self, N: int=24):
        """ Initialize the model. """
        self.N = N
        logger.debug(f"Initialized BaselineMostClicked with N={self.N}.")
    
    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, pd.DataFrame]):
        """ Fit the model to the data. """
        self.data = data
        logger.debug("Model fitted with provided data.")

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ Predict the most clicked items. """
        logger.debug(f"Predicting for user_id={user_id} at time={time} with top-k={k}.")
        time_window_start = time - pd.Timedelta(hours=self.N)
        behaviors_df = self.data['behaviors']
        behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'])
        filtered_behaviors_df = behaviors_df[
            (behaviors_df['Time'] >= time_window_start)
            & (behaviors_df['Time'] <= time)
        ]
        logger.debug(f"Filtered behaviors to {len(filtered_behaviors_df)} rows within the time window.")

        impressions_df = self.data['impressions']
        clicked_impressions_df = impressions_df[impressions_df['Clicked'] == 1]
        merged_clicked_impressions_filtered_behaviors_df = pd.merge(filtered_behaviors_df, clicked_impressions_df, on='Impression ID', how='left')
        most_clicked_news_id = merged_clicked_impressions_filtered_behaviors_df.groupby('News ID')['Clicked'].sum().sort_values(ascending=False).head(k).index.tolist()
        logger.debug(f"Predicted top-{k} most clicked news IDs: {most_clicked_news_id}.")
        return most_clicked_news_id

    def evaluate(self):
        """ TODO """
        logger.debug("Evaluation method called but not implemented.")


class BaselineCoOccurence(RecommenderSystem):
    """ TODO """


class BaselineRingBuffer(RecommenderSystem):
    """ TODO """
    
    def __init__(self):
        """ TODO """
        logger.debug("Initialized BaselineRingBuffer.")
    
    def fit(self, data: dict[str, pd.DataFrame]):
        """ Fit the model to the data. """
        self.data = data
        logger.debug("Model fitted with provided data.")
    
    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ TODO """
        logger.debug(f"Predicting for user_id={user_id} at time={time} with top-k={k}.")
    
    def evaluate(self):
        """ TODO """
        logger.debug("Evaluation method called but not implemented.")


if __name__ == "__main__":
    logger.info("Running tests for BaselineMostClicked...")
    
    # Load data
    from src.data_normalization import data_normalization
    data, embeddings = data_normalization(validation=False, try_load=True)
    logger.debug("Data and embeddings loaded successfully.")
    
    # Create model
    rs_most_clicked = BaselineMostClicked()
    rs_most_clicked.fit(data, embeddings)
    logger.debug("BaselineMostClicked model fitted.")

    # Predict
    N = 10
    user_ids = data["behaviors"]["User ID"].drop_duplicates().sample(n=N, random_state=42)
    logger.info(f"Predicting for {N} users...")

    # Random time in data["behaviors"]['Time']
    prediction_time = data['behaviors']['Time'].drop_duplicates().sample(n=1, random_state=42).values[0]
    prediction_time = pd.to_datetime(prediction_time)
    logger.debug(f"Selected prediction time: {prediction_time}.")

    for user_id in user_ids:
        logger.info(f"User ID: {user_id}")
        predictions = rs_most_clicked.predict(user_id, prediction_time)
        logger.info(f"Predictions: {predictions}")
