import pandas as pd
from scipy.sparse import csr_matrix, diags

from src.recommender_systems.collaborative_filtering import ItemItemCollaborativeFiltering
from src.recommender_systems.feature_based import ContentBasedFiltering
from src.recommender_systems import RecommenderSystem

class TrueHybrid(RecommenderSystem):
    """ TODO """
    
    def __init__(self, alpha: float = 0.5):
        """ TODO """
        super().__init__()
        self.alpha = alpha # Weight for combining predictions
        self.cf_model = ItemItemCollaborativeFiltering()
        self.cb_model = ContentBasedFiltering()
    
    def fit(self, data: dict[str, pd.DataFrame], embeddings: dict[str, pd.DataFrame]):
        # Train both collaborative and content-based models
        self.cf_model.fit(data, embeddings)
        self.cb_model.fit(data, embeddings)

        # Copy mappings from collaborative model (I'm assuming they use the same here)
        self.user_id_mapping = self.cf_model.user_id_mapping
        self.news_id_mapping = self.cf_model.news_id_mapping

    def predict(self, user_id: str, time: pd.Timestamp, k: int=10) -> list[str]:

        try:
            # Get top-100 recommendations from each model (many, to ensure overlap)
            cf_recs = self.cf_model.predict(user_id, time, k=100)
            cb_recs = self.cb_model.predict(user_id, time, k=100)

            # Convert ranked list into a score dictionary
            def rank_score(recs):
                return {item: len(recs) - i for i, item in enumerate(recs)}
            
            cf_scores = rank_score(cf_recs)
            cb_scores = rank_score(cb_recs)

            # Union of all items
            all_items = set(cf_scores.keys()).union(cb_scores.keys())

            # Comine scores using alpha - weighted average
            combined_scores = {
                item: self.alpha * cf_scores.get(item, 0) + (1 - self.alpha) * cb_scores.get(item, 0)
                for item in all_items
            }

            # Sort items by score
            top_k_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]

            # Return item IDs
            return [item for item, _ in top_k_items]
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.cf_model.predict(user_id, time, k)

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
        interactions_df['timestamp'] = interactions_df['Time'].apply(lambda x: int(x.timestamp()))
        interactions_df['timestamp'] = interactions_df['timestamp'].astype('int64')
        
        user_ids = interactions_df['User ID'].unique()
        news_ids = interactions_df['News ID'].unique()
        
        user_id_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        news_id_mapping = {news_id: idx for idx, news_id in enumerate(news_ids)}
        
        interactions_df['User Index'] = interactions_df['User ID'].map(user_id_mapping)
        interactions_df['News Index'] = interactions_df['News ID'].map(news_id_mapping)

        # For debugging
        print("timestamp dtype:", interactions_df['timestamp'].dtype)
        print("User Index dtype:", interactions_df['User Index'].dtype)
        print("News Index dtype:", interactions_df['News Index'].dtype)

        print("Any NaNs?")
        print("timestamp:", interactions_df['timestamp'].isna().sum())
        print("User Index:", interactions_df['User Index'].isna().sum())
        print("News Index:", interactions_df['News Index'].isna().sum())

        print("Example values:")
        print(interactions_df[['timestamp', 'User Index', 'News Index']].head(10))
        
        # Interaction matrix
        R = csr_matrix(
            (interactions_df['timestamp'], (interactions_df['User Index'], interactions_df['News Index'])),
            shape=(len(user_ids), len(news_ids))
        )

        return R, user_id_mapping, news_id_mapping

if __name__ == "__main__":
    print("Testing TrulyHybridRecommender...")

    from src.data_normalization import data_normalization
    data, embeddings = data_normalization(validation=False, try_load=True)

    hybrid = TrueHybrid(alpha=0.6)
    hybrid.fit(data, embeddings)

    user_ids = data["behaviors"]["User ID"].drop_duplicates().sample(n=5, random_state=42)
    for user_id in user_ids:
        print(f"\nUser: {user_id}")
        print(hybrid.predict(user_id, pd.Timestamp.now(), k=10))

        