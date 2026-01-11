import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Instant Music Recommender",
    layout="centered"
)

# Load trained model
nn_model, tfidf, df = pickle.load(open("model.pkl", "rb"))

st.title("🎵 Instant Music Recommender")

# Song selector
selected_song = st.selectbox(
    "🎶 Select a song:",
    df["song"].values
)

def recommend(song):
    index = df[df["song"] == song].index[0]

    song_text = df.iloc[index]["text"]
    song_vector = tfidf.transform([song_text])

    distances, indices = nn_model.kneighbors(song_vector)

    results = []
    for i in indices[0][1:]:
        results.append({
            "Artist": df.iloc[i]["artist"],
            "Song": df.iloc[i]["song"],
            "Link": df.iloc[i]["link"]
        })

    return pd.DataFrame(results)

if st.button("🚀 Recommend Similar Songs"):
    st.success("Top similar songs:")
    result = recommend(selected_song)
    result.index += 1
    st.table(result)
