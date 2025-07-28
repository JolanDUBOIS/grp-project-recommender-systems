import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import logger
from src.recommender_systems import RecommenderSystem


class ContentBasedFiltering(RecommenderSystem):
    """ TODO """
    
    def __init__(self):
        """ TODO """
        super().__init__()
        logger.debug("Initialized ContentBasedFiltering.")

    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, np.ndarray]):
        """ Fit the model to the data. """
        super().fit(data, embeddings)
        logger.debug("Starting model fitting.")
        # Get the user-item interaction matrix
        R, user_id_mapping, news_id_mapping = self.get_user_item_interaction_matrix(data)

        # Get news embeddings
        news_df = data['news'].copy()
        news_df["idx"] = news_df["News ID"].map(news_id_mapping)
        news_df = news_df.dropna(subset=["idx"])
        news_df["idx"] = news_df["idx"].astype(int)
        news_df.sort_values(by="idx", ascending=True, inplace=True)
        news_df["content"] = news_df["Title"] + " " + news_df["Abstract"]
        news_df["content"] = news_df["content"].fillna("")
        logger.debug("Prepared news content for TF-IDF vectorization.")

        vectorizer = TfidfVectorizer(max_features=20)
        article_embeddings = vectorizer.fit_transform(news_df["content"]).toarray()
        Sim = cosine_similarity(article_embeddings)
        logger.debug("Computed article embeddings and similarity matrix.")

        self.user_id_mapping = user_id_mapping
        self.news_id_mapping = news_id_mapping
        self.R = R
        self.Sim = Sim
        logger.debug("Model fitting completed.")

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ Predict top-k items for a user. """
        try:
            if self.R is None or self.Sim is None:
                raise ValueError("Model not trained. Call fit() first.")

            if user_id not in self.user_id_mapping:
                logger.debug(f"User ID {user_id} not found in mapping. Falling back to default prediction.")
                return super().predict(user_id, time, k)

            # Compute scores
            user_idx = self.user_id_mapping[user_id]
            user_vector = self.R.getrow(user_idx)
            product = user_vector.dot(self.Sim)
            scores = product.flatten()

            # Score items already seen by the user to -inf
            scores[user_vector.toarray().flatten() == 1] = -np.inf

            # Get top k items
            top_k_idx = scores.argsort()[::-1][:k]
            reverse_news_id_mapping = {idx: news_id for news_id, idx in self.news_id_mapping.items()}
            top_items = [reverse_news_id_mapping[idx] for idx in top_k_idx]
            logger.debug(f"Predicted top-{k} items for user {user_id}: {top_items}.")
            return top_items
        except Exception as e:
            logger.error(f"Error during prediction for user {user_id}: {e}")
            return super().predict(user_id, time, k)

    def evaluate(self):
        """ Evaluate the model on the data. """
        logger.debug("Evaluation method called but not implemented.")


if __name__ == "__main__":
    logger.info("Running tests for ContentBasedFiltering...")

    # Load data
    from src.data_normalization import data_normalization
    data, embeddings = data_normalization(validation=False, try_load=True)
    logger.debug("Data and embeddings loaded successfully.")

    # Create model
    rs = ContentBasedFiltering()

    # Fit model
    logger.info("Fitting model...")
    rs.fit(data, embeddings)

    # Predict
    N = 10
    user_ids = data["behaviors"]["User ID"].drop_duplicates().sample(n=N, random_state=42)
    logger.info(f"Predicting for {N} users...")

    for user_id in user_ids:
        logger.info(f"User ID: {user_id}")
        predictions = rs.predict(user_id, pd.Timestamp.now())
        logger.info(f"Predictions: {predictions}")
