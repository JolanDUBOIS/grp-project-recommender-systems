import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import logger
from src.recommender_systems import RecommenderSystem


class ALSMatrixFactorization(RecommenderSystem):
    """ Alternating Least Squares (ALS) Matrix Factorization for collaborative filtering. """
    
    LATENT_FACTORS = 10
    REGULARIZATION_PARAMS = {
        'lambda_U': 0.1,  # Regularization parameter for the user factors (L2 regularization)
        'lambda_V': 0.1  # Regularization parameter for the item factors (L1 regularization)
    }
    N_ITER = 20  # Number of iterations
    SPLIT = 10
    
    def __init__(self):
        """ Initialize the ALSMatrixFactorization model. """
        super().__init__()
        logger.debug("Initialized ALSMatrixFactorization.")

    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, np.ndarray]):
        """Fit the ALS model to the given data."""
        super().fit(data, embeddings)
        logger.debug("Starting model fitting.")
        self.R, self.user_id_mapping, self.news_id_mapping = self.get_user_item_interaction_matrix(data)
        
        self.U = np.random.normal(loc=0.0, scale=0.01, size=(len(self.user_id_mapping), self.LATENT_FACTORS))
        self.V = np.random.normal(loc=0.0, scale=0.01, size=(len(self.news_id_mapping), self.LATENT_FACTORS))
        logger.debug(f"Initialized U and V matrices with shapes {self.U.shape} and {self.V.shape}.")

        np.random.seed(42)
        users_indices = np.arange(len(self.user_id_mapping))
        items_indices = np.arange(len(self.news_id_mapping))
        np.random.shuffle(users_indices)
        np.random.shuffle(items_indices)
        split_users_indices = np.array_split(users_indices, self.SPLIT)
        split_items_indices = np.array_split(items_indices, self.SPLIT)

        for k in range(self.N_ITER):
            logger.info(f"Starting iteration {k + 1}/{self.N_ITER}.")
            split_number = k % self.SPLIT
            users_subset = split_users_indices[split_number]
            items_subset = split_items_indices[split_number]
            t1 = time.time()
            self.U = self._update_user_factors(self.U, self.V, self.R, users_subset)
            self.V = self._update_item_factors(self.U, self.V, self.R, items_subset)
            t2 = time.time()
            logger.info(f"Iteration {k + 1} completed in {int(t2 - t1)} seconds.")

        self.save(self.U, self.V)
        logger.info("Model fitting completed and saved.")

    def read_model(self, U_path: str, V_path: str):
        """ Load pre-trained user and item factor matrices from disk. """
        self.U = np.load(U_path)
        self.V = np.load(V_path)
        logger.info("Model loaded from disk. No need to fit.")

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ Predict top-k items for a given user at a specific time. """
        try:
            if self.U is None or self.V is None:
                raise ValueError("Model not trained. Call fit() first.")

            if user_id not in self.user_id_mapping:
                logger.debug(f"User ID {user_id} not found in mapping. Falling back to default prediction.")
                return super().predict(user_id, time, k)

            user_idx = self.user_id_mapping[user_id]
            user_predictions = np.dot(self.U[user_idx], self.V.T)
            user_items = self.R.getrow(user_idx).toarray().flatten()
            multiplier = 2

            while True:
                top_items_idx = np.argsort(user_predictions)[::-1][:multiplier * k]
                top_items_idx = [idx for idx in top_items_idx if user_items[idx] == 0]
                if len(top_items_idx) >= k:
                    top_k_idx = top_items_idx[:k]
                    reverse_news_id_mapping = {idx: news_id for news_id, idx in self.news_id_mapping.items()}
                    top_items = [reverse_news_id_mapping[idx] for idx in top_k_idx]
                    logger.debug(f"Predicted top-{k} items for user {user_id}: {top_items}.")
                    return top_items
                multiplier += 1
        except Exception as e:
            logger.error(f"Error during prediction for user {user_id}: {e}")
            return super().predict(user_id, time, k)
        
    def evaluate(self):
        """ Evaluate the performance of the ALS model. """
        logger.debug("Evaluation method called but not implemented.")

    def _update_user_factors(self, U: np.ndarray, V: np.ndarray, R: np.array, subset: np.ndarray=None) -> np.ndarray:
        """ Update user factors using ALS optimization. """
        logger.debug("Updating user factors.")
        if subset is None:
            subset = np.arange(U.shape[0])
        
        new_U = np.zeros_like(U)
        for i in tqdm(subset, desc="Updating user factors"):
            R_i = self.R.getrow(i).toarray().flatten()
            masked_V = V * R_i[:, np.newaxis]
            new_U[i, :] = np.linalg.solve(V.T @ masked_V + self.REGULARIZATION_PARAMS['lambda_U'] * np.eye(self.LATENT_FACTORS), V.T @ R_i)
        return new_U

    def _update_item_factors(self, U: np.ndarray, V: np.ndarray, R: np.array, subset: np.ndarray=None) -> np.ndarray:
        """ Update item factors using ALS optimization. """
        logger.debug("Updating item factors.")
        if subset is None:
            subset = np.arange(V.shape[0])

        new_V = np.zeros_like(V)
        for i in tqdm(subset, desc="Updating item factors"):
            R_i = self.R.getcol(i).toarray().flatten()
            masked_U = U * R_i[:, np.newaxis]
            new_V[i, :] = np.linalg.solve(U.T @ masked_U + self.REGULARIZATION_PARAMS['lambda_V'] * np.eye(self.LATENT_FACTORS), U.T @ R_i)
        return new_V
    
    def save(self, U: np.ndarray, V: np.ndarray):
        """ Save the user and item factor matrices to disk. """
        now = datetime.now().strftime("%Y-%m-%d")
        np.save(f"models/U_{now}.npy", U)
        np.save(f"models/V_{now}.npy", V)
        logger.info("Model saved to disk.")
