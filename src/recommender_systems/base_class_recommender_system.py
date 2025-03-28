from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


class RecommenderSystem(ABC):
    """ Abstract class for recommender systems. """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, np.ndarray]):
        """ Fit the model to the data. """
        pass

    @abstractmethod
    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ TODO """
        pass

    @abstractmethod
    def evaluate(self):   # TODO: Add parameters to this method
        """ Evaluate the model on the data. """
        pass

    @staticmethod
    def get_user_item_interaction_matrix(data: pd.DataFrame) -> tuple[csr_matrix, dict[str, int], dict[str, int]]:
        """ Get the user-item interaction matrix. """
        impressions_df = data['impressions']
        behaviors_df = data['behaviors']
        
        interactions_df = pd.merge(impressions_df, behaviors_df, on='Impression ID', how='left')
        interactions_df = interactions_df[['User ID', 'News ID', 'Clicked']]
        
        user_ids = interactions_df['User ID'].unique()
        news_ids = interactions_df['News ID'].unique()
        
        user_id_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        news_id_mapping = {news_id: idx for idx, news_id in enumerate(news_ids)}
        
        interactions_df['User Index'] = interactions_df['User ID'].map(user_id_mapping)
        interactions_df['News Index'] = interactions_df['News ID'].map(news_id_mapping)
        
        # Interaction matrix
        R = csr_matrix(
            (interactions_df['Clicked'], (interactions_df['User Index'], interactions_df['News Index'])),
            shape=(len(user_ids), len(news_ids))
        )

        return R, user_id_mapping, news_id_mapping
