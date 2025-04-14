from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
import warnings
import openai
import os

warnings.filterwarnings("ignore")
nltk.download('punkt')

# Hardcode your OpenAI API key (or retrieve it from an environment variable)
OPENAI_API_KEY = "you api key"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Configure Flask to serve static files from the "static" folder
app = Flask(__name__, static_folder="static", static_url_path="")

# Route to serve index.html from the static folder
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# --- Helper Functions ---
def load_ebay_data(file_path):
    """Loads the eBay product data from a CSV file, keeping all columns (including 'Main Image')."""
    try:
        df = pd.read_csv(r"you file path")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def tokenize_titles(df):
    """Tokenizes product titles for improved text processing."""
    df['Tokenized Title'] = df['Title'].astype(str).apply(lambda x: word_tokenize(x.lower()))
    return df

def preprocess_data(df):
    """
    Vectorizes product titles using TF-IDF and applies KMeans clustering.
    Returns the updated DataFrame, the KMeans model, and the vectorizer.
    """
    df = tokenize_titles(df)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Title'].astype(str))
    num_clusters = min(10, max(2, len(df) // 5))  # Dynamically decide number of clusters
    model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(X)
    return df, model, vectorizer

def find_similar_products(df, model, vectorizer, user_query, max_price=None):
    """
    Determines which cluster the user's query belongs to, filters the products in that cluster,
    and then computes cosine similarity (using TF-IDF) only on that subset.
    Optionally applies a max price filter.
    Returns the top 5 matching products.
    """
    # Transform the user's query
    query_vec = vectorizer.transform([user_query])
    
    # Predict the cluster label for the query
    query_cluster = model.predict(query_vec)[0]
    
    # Filter the dataframe to only include products in that cluster
    df_cluster = df[df['Cluster'] == query_cluster].copy()
    
    # Compute cosine similarities for the filtered data
    similarities = cosine_similarity(query_vec, vectorizer.transform(df_cluster['Title'].astype(str))).flatten()
    df_cluster['Similarity'] = similarities
    
    # If a max price is provided, filter the products by price
    if max_price is not None:
        df_cluster = df_cluster[df_cluster['Price'] <= max_price]
    
    # Sort products by similarity and drop duplicates by Title, then take top 5
    results = df_cluster.sort_values(by='Similarity', ascending=False).drop_duplicates(subset=['Title']).head(5)
    return results if not results.empty else None

def chat_with_openai(messages):
    """Sends a multi-turn conversation to OpenAI API and retrieves the response."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content",
                   "Sorry, I couldn't process your request at the moment.")
        else:
            return "Sorry, I couldn't process your request at the moment."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with OpenAI API: {e}"

# --- API Endpoints ---
@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint for general chatbot interaction.
    Expects JSON: { "message": "User message", "chat_history": [ ... ] }
    """
    data = request.get_json()
    user_message = data.get("message", "")
    # Retrieve chat history from client (or start with empty list)
    chat_history = data.get("chat_history", [])
    
    # Enforce the system message as the first entry (role context)
    system_message = {
        "role": "system",
        "content": "You are an eBay shopping assistant specializing in shoes. Your expertise is in finding, comparing, and providing detailed information about shoes available on eBay. You respond only to shoe-related questions such as styles, brands, sizes, prices, shipping details, reviews, and deals, and avoid discussing any products outside of footwear. Your tone is friendly, informative, and helpful. Use the content provided to guide users in making informed shoe purchases on eBay."
    }
    if not chat_history or chat_history[0].get("role") != "system":
        chat_history.insert(0, system_message)
    
    # Append the new user message
    chat_history.append({"role": "user", "content": user_message})
    
    # Get response from OpenAI
    response_text = chat_with_openai(chat_history)
    
    # Append assistant response to chat history
    chat_history.append({"role": "assistant", "content": response_text})
    
    return jsonify({"response": response_text, "chat_history": chat_history})

@app.route("/search", methods=["POST"])
def search():
    """
    Endpoint for product search.
    Expects JSON: { "query": "product query", "max_price": optional }
    """
    data = request.get_json()
    user_query = data.get("query", "")
    max_price = data.get("max_price")
    
    # Use the fixed Windows path to your dataset
    file_path = r"you file path" 
    
    df = load_ebay_data(file_path)
    if df is None:
        return jsonify({"error": "Data file not found"}), 404
    
    # Preprocess the data (TF-IDF vectorization and clustering)
    df, model, vectorizer = preprocess_data(df)
    
    # Find similar products using clustering to narrow the search scope
    results_df = find_similar_products(df, model, vectorizer, user_query, max_price)
    
    if results_df is None:
        return jsonify({"results": []})
    
    # Convert the results to a list of dictionaries and return
    results = results_df.to_dict(orient="records")
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
