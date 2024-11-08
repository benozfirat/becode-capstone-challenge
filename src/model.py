import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def find_similar_articles(query, data, article_embeddings, top_n=3, min_similarity=0.3):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], article_embeddings).flatten()

    
    data['similarity_score'] = similarities
    filtered_data = data[data['similarity_score'] >= min_similarity]

    top_articles = filtered_data.sort_values(by=['similarity_score', 'Publish_Date'], ascending=[False, False]).head(top_n)
    
    return top_articles[['title', 'summary', 'url', 'Publish_Date']]

def chatbot_response(query):
    articles = find_similar_articles(query, data, article_embeddings)
    if articles.empty:
        return "Sorry, I couldn't find any relevant articles."
    
    response = "Here are some articles that might answer your question:\n\n"
    for _, row in articles.iterrows():
        response += f"- **{row['title']}**\n  {row['summary']}\n  [Read more]({row['url']})\n  Published on: {row['Publish_Date']}\n\n"
    return response



data = pd.read_json('data/articles.json')
data = data.drop(['type'], axis=1)
data = data.dropna()
data = data.drop_duplicates()
data['Publish_Date'] = pd.to_datetime(data['Publish_Date'], utc=True)
data['text_to_embed'] = data['title'] + " " + data['summary']

model = SentenceTransformer('all-MiniLM-L12-v2')
embeddings_file = 'data/article_embeddings.npy'


try:
    article_embeddings = np.load(embeddings_file)
except FileNotFoundError:
    article_embeddings = model.encode(data['text_to_embed'].tolist(), show_progress_bar=True)
    np.save(embeddings_file, article_embeddings)

    with open('data/cleaned_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    with open('data/article_embeddings.pkl', 'wb') as f:
        pickle.dump(article_embeddings, f)

question = "Que peux-tu me dire sur Valence ?"
print(chatbot_response(question))
