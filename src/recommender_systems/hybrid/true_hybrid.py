import pandas as pd
from scipy.sparse import csr_matrix

from src.recommender_systems.collaborative_filtering import ItemItemCollaborativeFiltering
from src.recommender_systems.feature_based import ContentBasedFiltering
from src.recommender_systems import RecommenderSystem

from . import logger


class TrueHybrid(RecommenderSystem):
    """ True Hybrid Recommender combining collaborative and content-based filtering. """
    
    def __init__(self, alpha: float = 0.5):
        """ Initialize the TrueHybrid model with a weight for combining predictions. """
        super().__init__()
        self.alpha = alpha  # Weight for combining predictions
        self.cf_model = ItemItemCollaborativeFiltering()
        self.cb_model = ContentBasedFiltering()
        logger.debug(f"Initialized TrueHybrid with alpha={self.alpha}.")

    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, pd.DataFrame]):
        """ Train both collaborative and content-based models. """
        logger.debug("Fitting collaborative and content-based models.")
        self.cf_model.fit(data, embeddings)
        self.cb_model.fit(data, embeddings)

        # Copy mappings from collaborative model (assuming they use the same here)
        self.user_id_mapping = self.cf_model.user_id_mapping
        self.news_id_mapping = self.cf_model.news_id_mapping
        logger.debug("Model fitting completed.")

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ Predict top-k items for a user by combining collaborative and content-based scores. """
        try:
            logger.debug(f"Predicting for user_id={user_id} at time={time} with top-k={k}.")
            # Get top-100 recommendations from each model (many, to ensure overlap)
            cf_recs = self.cf_model.predict(user_id, time, k=100)
            cb_recs = self.cb_model.predict(user_id, time, k=100)

            # Convert ranked list into a score dictionary
            def rank_score(recs):
                return {item: len(recs) - i for i, item in enumerate(recs)}
            
            cf_scores = rank_score(cf_recs)
            cb_scores = rank_score(cb_recs)

            # Union of all items
            all_items = set(cf_scores.keys()).union(cb_scores.keys())

            # Combine scores using alpha - weighted average
            combined_scores = {
                item: self.alpha * cf_scores.get(item, 0) + (1 - self.alpha) * cb_scores.get(item, 0)
                for item in all_items
            }

            # Sort items by score
            top_k_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]

            # Return item IDs
            predictions = [item for item, _ in top_k_items]
            logger.debug(f"Predicted top-{k} items for user {user_id}: {predictions}.")
            return predictions
        
        except Exception as e:
            logger.error(f"Prediction error for user {user_id}: {e}")
            return self.cf_model.predict(user_id, time, k)

    def evaluate(self):
        """ Evaluate the performance of the True Hybrid model. """
        logger.debug("Evaluation method called but not implemented.")

    @staticmethod
    def get_user_item_interaction_matrix(data: pd.DataFrame) -> tuple[csr_matrix, dict[str, int], dict[str, int]]:
        """ Generate the user-item interaction matrix. """
        logger.debug("Generating user-item interaction matrix.")
        impressions_df = data['impressions']
        behaviors_df = data['behaviors']
        
        interactions_df = pd.merge(impressions_df, behaviors_df, on='Impression ID', how='left')
        interactions_df = interactions_df[['User ID', 'News ID', 'Clicked', 'Time']]
        interactions_df['Time'] = pd.to_datetime(interactions_df['Time'])
        interactions_df['timestamp'] = interactions_df['Time'].apply(lambda x: int(x.timestamp()))
        interactions_df['timestamp'] = interactions_df['timestamp'].astype('int64')
        
        user_ids = interactions_df['User ID'].unique()
        news_ids = interactions_df['News ID'].unique()
        
        user_id_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        news_id_mapping = {news_id: idx for idx, news_id in enumerate(news_ids)}
        
        interactions_df['User Index'] = interactions_df['User ID'].map(user_id_mapping)
        interactions_df['News Index'] = interactions_df['News ID'].map(news_id_mapping)

        # For debugging
        logger.debug(f"timestamp dtype: {interactions_df['timestamp'].dtype}")
        logger.debug(f"User Index dtype: {interactions_df['User Index'].dtype}")
        logger.debug(f"News Index dtype: {interactions_df['News Index'].dtype}")

        logger.debug("Any NaNs?")
        logger.debug(f"timestamp: {interactions_df['timestamp'].isna().sum()}")
        logger.debug(f"User Index: {interactions_df['User Index'].isna().sum()}")
        logger.debug(f"News Index: {interactions_df['News Index'].isna().sum()}")

        logger.debug("Example values:")
        logger.debug(f"{interactions_df[['timestamp', 'User Index', 'News Index']].head(10)}")

        # Interaction matrix
        R = csr_matrix(
            (interactions_df['timestamp'], (interactions_df['User Index'], interactions_df['News Index'])),
            shape=(len(user_ids), len(news_ids))
        )
        logger.debug("User-item interaction matrix generated.")
        return R, user_id_mapping, news_id_mapping


if __name__ == "__main__":
    logger.info("Testing TrueHybrid...")

    from src.data_normalization import data_normalization
    data, embeddings = data_normalization(validation=False, try_load=True)
    logger.debug("Data and embeddings loaded successfully.")

    hybrid = TrueHybrid(alpha=0.6)
    logger.info("Fitting TrueHybrid model...")
    hybrid.fit(data, embeddings)
    logger.info("Model fitted.")

    user_ids = data["behaviors"]["User ID"].drop_duplicates().sample(n=5, random_state=42)
    logger.info("Predicting for 5 users...")
    for user_id in user_ids:
        logger.info(f"User ID: {user_id}")
        predictions = hybrid.predict(user_id, pd.Timestamp.now(), k=10)
        logger.info(f"Predictions: {predictions}")

