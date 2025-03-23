import time
from datetime import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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

    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, np.ndarray]):
        """ Fit the model to the data. """
        # print(f"Type data: {type(data)}")
        self.data = data
        behaviors_df = self.data['behaviors']
        news_df = self.data['news']
        impressions_df = self.data['impressions']

        self.users_ids = behaviors_df['User ID'].unique()
        self.news_ids = news_df['News ID'].unique()
        
        user_id_mapping = {user_id: idx for idx, user_id in enumerate(self.users_ids)}
        news_id_mapping = {news_id: idx for idx, news_id in enumerate(self.news_ids)}
        
        interactions_df = pd.merge(impressions_df, behaviors_df, on='Impression ID', how='left')
        interactions_df = interactions_df[['User ID', 'News ID', 'Clicked']]

        interactions_df['User Index'] = interactions_df['User ID'].map(user_id_mapping)
        interactions_df['News Index'] = interactions_df['News ID'].map(news_id_mapping)
        
        R = csr_matrix(
            (interactions_df['Clicked'], (interactions_df['User Index'], interactions_df['News Index'])),
            shape=(len(self.users_ids), len(self.news_ids))
        )
        
        U = np.random.normal(loc=0.0, scale=0.01, size=(len(self.users_ids), self.LATENT_FACTORS))
        V = np.random.normal(loc=0.0, scale=0.01, size=(len(self.news_ids), self.LATENT_FACTORS))
        
        np.random.seed(42)
        users_indices = np.arange(len(self.users_ids))
        items_indices = np.arange(len(self.news_ids))
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
            U = self._update_user_factors(U, V, R, users_subset)
            V = self._update_item_factors(U, V, R, items_subset)
            t2 = time.time()
            print(f"Iteration took {int(t2 - t1)}s.")
        
        self.save(U, V)

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ TODO """
        if self.U is None or self.V is None:
            raise ValueError("Model not trained. Call fit() first.")
        user_predictions = np.dot(self.U[user_id], self.V.T)
        top_items = np.argsort(user_predictions)[::-1][:k]
        return top_items
        
    def evaluate(self):
        """ TODO """

    def _update_user_factors(self, U: np.ndarray, V: np.ndarray, R: csr_matrix, subset: np.ndarray=None) -> np.ndarray:
        """ TODO """
        print("Update user factors")
        if subset is None:
            subset = np.arange(U.shape[0])
        
        new_U = np.zeros_like(U)
        for i in tqdm(subset):
            R_i = R.getrow(i).toarray().flatten()
            masked_V = V * R_i[:, np.newaxis]
            new_U[i, :] = np.linalg.solve(V.T @ masked_V + self.REGULARIZATION_PARAMS['lambda_U'] * np.eye(self.LATENT_FACTORS), V.T @ R_i)
        return new_U

    def _update_item_factors(self, U: np.ndarray, V: np.ndarray, R: csr_matrix, subset: np.ndarray=None) -> np.ndarray:
        """ TODO """
        print("Update item factors")
        if subset is None:
            subset = np.arange(V.shape[0])

        new_V = np.zeros_like(V)
        for i in tqdm(subset):
            R_i = R.getcol(i).toarray().flatten()
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
