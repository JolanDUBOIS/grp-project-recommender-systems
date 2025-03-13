from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from src.main import data_normalization

# Load normalized data
data, embeddings = data_normalization()

# Extract relevant dataframes
news_df = data["news"]
impressions_df = data["impressions"]

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2') # This is a rather lightweight model

# To start, use titles and abstracts
news_df["content"] = news_df["Title"] + " " + news_df["Abstract"]

# Convert articles to embeddings
article_embeddings = model.encode(news_df["content"].tolist(), convert_to_tensor=True)

# Compute similarity matrix
similarity_matrix = cosine_similarity(article_embeddings.cpu().numpy())

# Store similarity results
news_similarity_df = pd.DataFrame(similarity_matrix, index=news_df["News ID"], columns=news_df["News ID"])
