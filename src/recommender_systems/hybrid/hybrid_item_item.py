import pandas as pd
from scipy.sparse import csr_matrix, diags

from . import logger
from src.recommender_systems.collaborative_filtering import ItemItemCollaborativeFiltering


class HybridItemItemCollabFiltering(ItemItemCollaborativeFiltering):
    """ Hybrid Item-Item Collaborative Filtering with time decay. """
    
    def __init__(self):
        """ Initialize the HybridItemItemCollabFiltering model. """
        super().__init__()
        logger.debug("Initialized HybridItemItemCollabFiltering.")

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ Predict top-k items for a user with time-decayed interactions. """
        logger.debug(f"Predicting for user_id={user_id} at time={time} with top-k={k}.")
        timestp = int(time.timestamp())
        self.R.data = abs(self.R.data - timestp) / (3600 * 24)
        self.R.data = 1 / self.R.data
        logger.debug("Adjusted interaction matrix with time decay.")
        return super().predict(user_id, time, k)

    def evaluate(self):
        """ Evaluate the performance of the Hybrid Item-Item model. """
        logger.debug("Evaluation method called but not implemented.")

    @staticmethod
    def get_user_item_interaction_matrix(data: pd.DataFrame) -> tuple[csr_matrix, dict[str, int], dict[str, int]]:
        """ Generate the user-item interaction matrix with time decay. """
        logger.debug("Generating user-item interaction matrix.")
        impressions_df = data['impressions']
        behaviors_df = data['behaviors']
        
        interactions_df = pd.merge(impressions_df, behaviors_df, on='Impression ID', how='left')
        interactions_df = interactions_df[['User ID', 'News ID', 'Clicked', 'Time']]
        interactions_df['Time'] = pd.to_datetime(interactions_df['Time'])
        interactions_df['timestamp'] = interactions_df['Time'].apply(
            lambda x: int(x.timestamp()) if pd.notnull(x) else 0
        )

        user_ids = interactions_df['User ID'].unique()
        news_ids = interactions_df['News ID'].unique()
        
        user_id_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        news_id_mapping = {news_id: idx for idx, news_id in enumerate(news_ids)}
        
        interactions_df['User Index'] = interactions_df['User ID'].map(user_id_mapping)
        interactions_df['News Index'] = interactions_df['News ID'].map(news_id_mapping)
        
        # Interaction matrix
        R = csr_matrix(
            (interactions_df['timestamp'] * interactions_df['Clicked'], (interactions_df['User Index'], interactions_df['News Index'])),
            shape=(len(user_ids), len(news_ids))
        )
        logger.debug("User-item interaction matrix generated.")
        return R, user_id_mapping, news_id_mapping


if __name__ == "__main__":
    logger.info("Running tests for HybridItemItemCollabFiltering...")

    # Load data
    from src.data_normalization import data_normalization
    data, embeddings = data_normalization(validation=False, try_load=True)
    logger.debug("Data and embeddings loaded successfully.")

    # Create model
    rs_hybrid = HybridItemItemCollabFiltering()
    rs_item_item = ItemItemCollaborativeFiltering()

    # Fit model
    logger.info("Fitting models...")
    rs_hybrid.fit(data, embeddings)
    rs_item_item.fit(data, embeddings)
    logger.info("Models fitted.")

    # Predict
    N = 10
    user_ids = data["behaviors"]["User ID"].drop_duplicates().sample(n=N, random_state=42)
    logger.info(f"Predicting for {N} users...")

    for user_id in user_ids:
        logger.info(f"User ID: {user_id}")
        hybrid_predictions = rs_hybrid.predict(user_id, pd.Timestamp.now())
        logger.info(f"Hybrid Item-Item Collaborative Filtering Predictions: {hybrid_predictions}")
        item_item_predictions = rs_item_item.predict(user_id, pd.Timestamp.now())
        logger.info(f"Item-Item Collaborative Filtering Predictions: {item_item_predictions}")

