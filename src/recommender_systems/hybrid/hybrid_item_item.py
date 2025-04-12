import pandas as pd
from scipy.sparse import csr_matrix, diags

from src.recommender_systems.collaborative_filtering import ItemItemCollaborativeFiltering


class HybridItemItemCollabFiltering(ItemItemCollaborativeFiltering):
    """ TODO """
    
    def __init__(self):
        """ TODO """
        super().__init__()
    
    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:
        timestp = int(time.timestamp())
        self.R.data = abs(self.R.data - timestp) / (3600 * 24)
        self.R.data = 1 / self.R.data
        return super().predict(user_id, time, k)

    def evaluate(self):
        """ Evaluate the model on the data. """
        pass

    @staticmethod
    def get_user_item_interaction_matrix(data: pd.DataFrame) -> tuple[csr_matrix, dict[str, int], dict[str, int]]:
        """ Get the user-item interaction matrix. """
        impressions_df = data['impressions']
        behaviors_df = data['behaviors']
        
        interactions_df = pd.merge(impressions_df, behaviors_df, on='Impression ID', how='left')
        interactions_df = interactions_df[['User ID', 'News ID', 'Clicked', 'Time']]
        interactions_df['Time'] = pd.to_datetime(interactions_df['Time'])
        interactions_df['timestamp'] = interactions_df['Time'].apply(
            lambda x: int(x.timestamp()) if pd.notnull(x) else 0
        )

        user_ids = interactions_df['User ID'].unique()
        news_ids = interactions_df['News ID'].unique()
        
        user_id_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        news_id_mapping = {news_id: idx for idx, news_id in enumerate(news_ids)}
        
        interactions_df['User Index'] = interactions_df['User ID'].map(user_id_mapping)
        interactions_df['News Index'] = interactions_df['News ID'].map(news_id_mapping)
        
        # Interaction matrix
        R = csr_matrix(
            (interactions_df['timestamp']*interactions_df['Clicked'], (interactions_df['User Index'], interactions_df['News Index'])),
            shape=(len(user_ids), len(news_ids))
        )

        return R, user_id_mapping, news_id_mapping

if __name__ == "__main__":
    print("Running tests for HybridItemItemCollabFiltering...")

    # Load data
    from src.data_normalization import data_normalization
    data, embeddings = data_normalization(validation=False, try_load=True)

    # Create model
    rs_hybrid = HybridItemItemCollabFiltering()
    rs_item_item = ItemItemCollaborativeFiltering()

    # Fit model
    print("Fitting model...")
    rs_hybrid.fit(data, embeddings)
    rs_item_item.fit(data, embeddings)
    print("Model fitted.")

    # Predict
    N = 10
    user_ids = data["behaviors"]["User ID"].drop_duplicates().sample(n=N, random_state=42)

    print(f"Predicting for {N} users...")
    for user_id in user_ids:
        print(f"User ID: {user_id}")
        print("Hybrid Item-Item Collaborative Filtering:")
        print(rs_hybrid.predict(user_id, pd.Timestamp.now()))
        print("Item-Item Collaborative Filtering:")
        print(rs_item_item.predict(user_id, pd.Timestamp.now()))
        print()

