from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from . import logger


class RecommenderSystem(ABC):
    """ Abstract class for recommender systems. """

    def __init__(self):
        logger.debug("Initialized RecommenderSystem base class.")

    @abstractmethod
    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, np.ndarray]):
        """ Fit the model to the data. """
        logger.debug("Called abstract fit method.")
        self.data = data
        self.embeddings = embeddings

    @abstractmethod
    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ Generate a list of recommended items for a given user. """
        logger.debug(f"Called abstract predict method for user_id={user_id}, time={time}, k={k}.")
        return self.data["impressions"][self.data["impressions"]["Clicked"] == 1]["News ID"].value_counts().head(k).index.tolist()  # Most clicked

    @abstractmethod
    def evaluate(self):
        """ Evaluate the model on the data. """
        logger.debug("Called abstract evaluate method.")
        pass

    @staticmethod
    def get_user_item_interaction_matrix(data: pd.DataFrame) -> tuple[csr_matrix, dict[str, int], dict[str, int]]:
        """ Get the user-item interaction matrix. """
        logger.debug("Generating user-item interaction matrix.")
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

        logger.debug("User-item interaction matrix generated successfully.")
        return R, user_id_mapping, news_id_mapping
