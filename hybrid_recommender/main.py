import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from src.main import data_normalization


# Load the normalized data
data, embeddings = data_normalization()

# Extract relevant dataframes
news_df = data["news"]
impressions_df = data["impressions"]

# Ensure required columns exist
if "Clicked" in impressions_df.columns and "User ID" in impressions_df.columns and "News ID" in impressions_df.columns:
    impressions_df["Clicked"] = impressions_df["Clicked"].astype(int)
    interaction_matrix = impressions_df.pivot(index="User ID", columns="News ID", values="Clicked").filla(0)
else:
    interaction_matrix = pd.DataFrame()

# Create User-Item Interaction matrix
impressions_df["Clicked"] = impressions_df["Clicked"].astype(int)
interaction_matrix = impressions_df.pivot(index="User ID", columns="News ID", values="Clicked").fillna(0)

# Apply collaborative filtering
U, sigma, Vt = svds(interaction_matrix, k=50) # Reduce dimensions
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_df = pd.DataFrame(predicted_ratings, index=interaction_matrix.index, columns=interaction_matrix.columns)


# Apply content-based filtering
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(news_df["Title"] + " " + news_df["Abstract"])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Store results
news_similarity_df = pd.DataFrame(similarity_matrix, index=news_df["News ID"], columns=news_df["News ID"])

# Hybrid approach, combining scores
def hybrid_recommendations(user_id, top_n=5, alpha=0.5):
    if user_id not in predicted_df.index:
        return [] # No recommendations for uknown/new users
    
    user_ratings = predicted_df.loc[user_id].copy()
    user_ratings = (user_ratings - user_ratings.min()) / (user_ratings.max() - user_ratings.min()) # Normalize

    content_scores = news_similarity_df[user_ratings.index].dot(user_ratings.fillna(0))
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min()) # Normalize

    final_scores = alpha * user_ratings + (1 - alpha) * content_scores
    return final_scores.nlargest(top_n).index.tolist()

# Test
user_id = interaction_matrix.index[0] # Pick a sample user
recommendations = hybrid_recommendations(user_id)
print("Recommended articles: ", recommendations)