import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.recommender_systems import RecommenderSystem


class ItemItemCollaborativeFiltering(RecommenderSystem):
    """ TODO """

    def __init__(self):
        """ TODO """

    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, np.ndarray]):
        """ Fit the model to the data. """
        # Get the user-item interaction matrix
        R, user_id_mapping, news_id_mapping = self.get_user_item_interaction_matrix(data)

        # Get item-item similarity matrix (cosine similarity)
        Sim = self.cosine_similarity(R)

        self.user_id_mapping = user_id_mapping
        self.news_id_mapping = news_id_mapping
        self.R = R
        self.Sim = Sim

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ TODO """
        if self.R is None or self.Sim is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Compute scores
        user_idx = self.user_id_mapping[user_id]
        user_vector = self.R.getrow(user_idx)
        product = user_vector.dot(self.Sim)
        scores = product.toarray().flatten()

        # Score items already seen by the user to -inf
        scores[user_vector.toarray().flatten() == 1] = -np.inf

        # Get top k items
        top_k_idx = scores.argsort()[::-1][:k]
        reverse_news_id_mapping = {idx: news_id for news_id, idx in self.news_id_mapping.items()}
        top_items = [reverse_news_id_mapping[idx] for idx in top_k_idx]

        return top_items

    def evaluate(self):
        """ Evaluate the model on the data. """
        pass

    @staticmethod
    def cosine_similarity(R: csr_matrix, axis: int=0) -> csr_matrix:
        """ TODO """
        norms = np.sqrt(R.power(2).sum(axis=axis))
        norms = np.array(norms).flatten()
        norms[norms == 0] = 1 # Avoid division
        similarity = R.T @ R
        similarity = similarity.multiply(1 / norms).multiply(1 / norms.T)
        similarity = csr_matrix(similarity)
        return similarity

if __name__ == "__main__":
    print("Running tests for ItemItemCollaborativeFiltering...")

    # Load data
    from src.data_normalization import data_normalization
    data, embeddings = data_normalization(validation=False, try_load=True)

    # Create model
    rs = ItemItemCollaborativeFiltering()

    # Fit model
    print("Fitting model...")
    rs.fit(data, embeddings)

    # Predict
    N = 10
    user_ids = data["behaviors"]["User ID"].drop_duplicates().sample(n=N, random_state=42)

    print(f"Predicting for {N} users...")
    for user_id in user_ids:
        print(f"User ID: {user_id}")
        print(rs.predict(user_id, pd.Timestamp.now()))
        print()
