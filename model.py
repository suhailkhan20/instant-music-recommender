import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

# Load dataset
df = pd.read_csv("music.csv")
df.columns = df.columns.str.lower()

# Handle missing text
df["text"] = df["text"].fillna("")

# Convert text to TF-IDF vectors
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["text"])

# Train Nearest Neighbors model
nn_model = NearestNeighbors(
    n_neighbors=6,
    metric="cosine",
    algorithm="brute"
)
nn_model.fit(tfidf_matrix)

# Save model and vectorizer
pickle.dump((nn_model, tfidf, df), open("model.pkl", "wb"))

print("✅ NearestNeighbors model saved successfully")
