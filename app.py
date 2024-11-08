import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Charger les embeddings et les articles depuis pickle
@st.cache_resource  # Mise en cache pour éviter de recharger à chaque fois
def load_embeddings_and_articles():
    with open(r'data/article_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    with open(r'data/cleaned_data.pkl', 'rb') as f:
        articles = pickle.load(f)
    return embeddings, articles


embeddings, articles = load_embeddings_and_articles()


st.title("Article Search")

search = st.text_input("What are you looking for?")

if search:
    model = SentenceTransformer('all-MiniLM-L12-v2')
    query_embedding = model.encode([search])[0]
    similarities = cosine_similarity([query_embedding], embeddings).flatten()

    articles['similarity_score'] = similarities
    filtered_articles = articles[articles['similarity_score'] >= 0.3]
    top_articles = filtered_articles.sort_values(by=['similarity_score', 'Publish_Date'], ascending=[False, False]).head(3)

    if top_articles.empty:
        st.write("No relevant articles found.")
    else:
        for _, row in top_articles.iterrows():
            st.write(f"**{row['title']}**")
            st.write(row['summary'])
            st.write(f"[Read more]({row['url']})")
            st.write(f"Published on: {row['Publish_Date']}")
            st.write("----")


