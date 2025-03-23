from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


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
