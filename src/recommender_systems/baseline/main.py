import pandas as pd

from src.recommender_systems import RecommenderSystem


class BaselineMostClicked(RecommenderSystem):
    """ Baseline recommender system. """
    # TODO - Rename most popular (cf lecture 06/02)
    # This function is a most popular with a time window which looks a bit like ring buffer but not really
    
    def __init__(self, N: int=24):
        """ Initialize the model. """
        self.N = N
    
    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, pd.DataFrame]):
        """ Fit the model to the data. """
        self.data = data

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ TODO """
        time_window_start = time - pd.Timedelta(hours=self.N)
        behaviors_df = self.data['behaviors']
        behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'])
        filtered_behaviors_df = behaviors_df[
            (behaviors_df['Time'] >= time_window_start)
            & (behaviors_df['Time'] <= time)
        ]
        impressions_df = self.data['impressions']
        clicked_impressions_df = impressions_df[impressions_df['Clicked'] == 1]
        merged_clicked_impressions_filtered_behaviors_df = pd.merge(filtered_behaviors_df, clicked_impressions_df, on='Impression ID', how='left')
        most_clicked_news_id = merged_clicked_impressions_filtered_behaviors_df.groupby('News ID')['Clicked'].sum().sort_values(ascending=False).head(k).index.tolist()
        # TODO - switch to clicked rate + news that just got out ? 
        return most_clicked_news_id

    def evaluate(self):
        """ TODO """

class BaselineCoOccurence(RecommenderSystem):
    """ TODO """

class BaselineRingBuffer(RecommenderSystem):
    """ TODO """
    
    def __init__(self):
        """ TODO """
    
    def fit(self, data: dict[str, pd.DataFrame]):
        """ Fit the model to the data. """
        self.data = data
    
    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        """ TODO """
    
    def evaluate(self):
        """ TODO """
    