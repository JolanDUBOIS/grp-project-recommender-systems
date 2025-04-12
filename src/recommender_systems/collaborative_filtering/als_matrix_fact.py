import time
from datetime import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd

from src.recommender_systems import RecommenderSystem


class ALSMatrixFactorization(RecommenderSystem):
    """ TODO """
    
    LATENT_FACTORS = 10
    REGULARIZATION_PARAMS = {
        'lambda_U': 0.1,  # Regularization parameter for the user factors (L2 regularization)
        'lambda_V': 0.1  # Regularization parameter for the item factors (L1 regularization)
        # TODO - Grid search or optima to find the best parameters
    }
    N_ITER = 10  # Number of iterations
    SPLIT = 10
    
    def __init__(self):
        """ TODO """
        super().__init__()

    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, np.ndarray]):
        """ Fit the model to the data. """
        super().fit(data, embeddings)
        self.R, self.user_id_mapping, self.news_id_mapping = self.get_user_item_interaction_matrix(data)
        
        self.U = np.random.normal(loc=0.0, scale=0.01, size=(len(self.user_id_mapping), self.LATENT_FACTORS))
        self.V = np.random.normal(loc=0.0, scale=0.01, size=(len(self.news_id_mapping), self.LATENT_FACTORS))
        
        np.random.seed(42)
        users_indices = np.arange(len(self.user_id_mapping))
        items_indices = np.arange(len(self.news_id_mapping))
        np.random.shuffle(users_indices)
        np.random.shuffle(items_indices)
        split_users_indices = np.array_split(users_indices, self.SPLIT)
        split_items_indices = np.array_split(items_indices, self.SPLIT)

        for k in range(self.N_ITER):
            print(f"Iteration {k} / {self.N_ITER}")
            split_number = k % self.SPLIT
            users_subset = split_users_indices[split_number]
            items_subset = split_items_indices[split_number]
            t1 = time.time()
            self.U = self._update_user_factors(self.U, self.V, self.R, users_subset)
            self.V = self._update_item_factors(self.U, self.V, self.R, items_subset)
            t2 = time.time()
            print(f"Iteration took {int(t2 - t1)}s.")

        self.save(self.U, self.V)  # Save the model (temporary)

    def read_model(self, U_path: str, V_path: str):
        """ Read a model """
        self.U = np.load(U_path)
        self.V = np.load(V_path)
        print("Model loaded, no need to fit.")

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ TODO """
        try:
            if self.U is None or self.V is None:
                raise ValueError("Model not trained. Call fit() first.")

            # Get the user index
            user_idx = self.user_id_mapping[user_id]

            # Compute the predictions
            user_predictions = np.dot(self.U[user_idx], self.V.T)

            user_items = self.R.getrow(user_idx).toarray().flatten()
            multiplier = 2
            while True:
                # Get the top 2*k items
                top_items_idx = np.argsort(user_predictions)[::-1][:multiplier*k]

                # Filter out items already seen by the user
                top_items_idx = [idx for idx in top_items_idx if user_items[idx] == 0]
                if len(top_items_idx) >= k:
                    top_k_idx = top_items_idx[:k]

                    # Get the news ids
                    reverse_news_id_mapping = {idx: news_id for news_id, idx in self.news_id_mapping.items()}
                    top_items = [reverse_news_id_mapping[idx] for idx in top_k_idx]

                    return top_items

                multiplier += 1
        except Exception as e:
            print(f"Error in prediction: {e}")
            return super().predict(user_id, time, k)
        
    def evaluate(self):
        """ TODO """

    def _update_user_factors(self, U: np.ndarray, V: np.ndarray, subset: np.ndarray=None) -> np.ndarray:
        """ TODO """
        print("Update user factors")
        if subset is None:
            subset = np.arange(U.shape[0])
        
        new_U = np.zeros_like(U)
        for i in tqdm(subset):
            R_i = self.R.getrow(i).toarray().flatten()
            masked_V = V * R_i[:, np.newaxis]
            new_U[i, :] = np.linalg.solve(V.T @ masked_V + self.REGULARIZATION_PARAMS['lambda_U'] * np.eye(self.LATENT_FACTORS), V.T @ R_i)
        return new_U

    def _update_item_factors(self, U: np.ndarray, V: np.ndarray, subset: np.ndarray=None) -> np.ndarray:
        """ TODO """
        print("Update item factors")
        if subset is None:
            subset = np.arange(V.shape[0])

        new_V = np.zeros_like(V)
        for i in tqdm(subset):
            R_i = self.R.getcol(i).toarray().flatten()
            masked_U = U * R_i[:, np.newaxis]
            new_V[i, :] = np.linalg.solve(U.T @ masked_U + self.REGULARIZATION_PARAMS['lambda_V'] * np.eye(self.LATENT_FACTORS), U.T @ R_i)
        return new_V
    
    def save(self, U: np.ndarray, V: np.ndarray):
        """ Save the model """
        now = datetime.now().strftime("%Y-%m-%d")
        np.save(f"models/U_{now}.npy", U)
        np.save(f"models/V_{now}.npy", V)
        print("Model saved.")

# TODO - To make the computation faster, we want to separate the users and items into small groups (time ?)
