import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Instant Music Recommender", layout="centered")

st.title("🎵 Instant Music Recommender")
st.caption("Content-based music recommendation using TF-IDF & Nearest Neighbors")

@st.cache_data
def load_data():
    df = pd.read_csv("music.csv")
    df.columns = df.columns.str.lower()
    df["text"] = df["text"].fillna("")
    return df

@st.cache_resource
def train_model(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["text"])

    nn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    nn.fit(tfidf_matrix)

    return tfidf, nn, tfidf_matrix

df = load_data()
tfidf, nn_model, tfidf_matrix = train_model(df)

selected_song = st.selectbox("🎶 Select a song:", df["song"].values)

def recommend(song):
    idx = df[df["song"] == song].index[0]
    song_vec = tfidf_matrix[idx]
    distances, indices = nn_model.kneighbors(song_vec)

    results = []
    for i in indices[0][1:]:
        results.append({
            "Artist": df.iloc[i]["artist"],
            "Song": df.iloc[i]["song"],
            "Link": df.iloc[i]["link"]
        })

    return pd.DataFrame(results)

if st.button("🎧 Recommend Similar Songs"):
    result = recommend(selected_song)

    if result.empty:
        st.warning("No recommendations found.")
    else:
        st.success("Top similar songs:")
        st.dataframe(result, use_container_width=True)


