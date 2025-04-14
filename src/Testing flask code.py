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

# Hardcoded OpenAI API key and endpoint
OPENAI_API_KEY = "you api key"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Configure Flask to serve static files from the "static" folder
app = Flask(__name__, static_folder="static", static_url_path="")

# Serve the index.html file at the root route
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# --- Functions from your original Python code ---
def load_ebay_data(file_path):
    """Loads the eBay product data from a CSV file."""
    try:
        # Update the path to your CSV file as needed.
        return pd.read_csv("your file path")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def tokenize_titles(df):
    """Tokenizes product titles for improved text processing."""
    df['Tokenized Title'] = df['Title'].astype(str).apply(lambda x: word_tokenize(x.lower()))
    return df

def preprocess_data(df):
    """Vectorizes product titles using TF-IDF and clusters them."""
    df = tokenize_titles(df)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Title'].astype(str))
    num_clusters = min(10, max(2, len(df) // 5))  # Ensure at least 2 clusters
    model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(X)
    return df, model, vectorizer

def find_similar_products(df, model, vectorizer, user_query, max_price=None):
    """
    Finds the most similar product titles based on user input using cosine similarity.
    Returns unique products and applies a max price filter if provided.
    """
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, vectorizer.transform(df['Title'].astype(str))).flatten()
    df['Similarity'] = similarities
    if max_price is not None:
        df = df[df['Price'] <= max_price]
    results = df.sort_values(by='Similarity', ascending=False).drop_duplicates(subset=['Title']).head(5)
    return results if not results.empty else None

def chat_with_openai(messages):
    """Sends multi-turn conversation to OpenAI API and retrieves response."""
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
    chat_history = data.get("chat_history", [])
    
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
    Expects JSON: { "query": "product query", "max_price": optional, "file_path": "path/to/csv" }
    """
    data = request.get_json()
    user_query = data.get("query", "")
    max_price = data.get("max_price")
    file_path = data.get("file_path", "you file path")
    
    df = load_ebay_data(file_path)
    if df is None:
        return jsonify({"error": "Data file not found"}), 404
    
    df, model, vectorizer = preprocess_data(df)
    results_df = find_similar_products(df, model, vectorizer, user_query, max_price)
    
    if results_df is None:
        return jsonify({"results": []})
    
    # Convert DataFrame rows to list of dicts for JSON response
    results = results_df.to_dict(orient="records")
    return jsonify({"results": results})

if __name__ == "__main__":
    # Run the app on localhost:5000 with debug mode enabled
    app.run(debug=True)
