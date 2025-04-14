import os
import warnings

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import requests
import nltk
from nltk.tokenize import word_tokenize
import openai

# Semantic + Clustering + FAISS
from sentence_transformers import SentenceTransformer
import hdbscan
import faiss

from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
nltk.download('punkt')

# --- CONFIGURATION ---
OPENAI_API_KEY = "your api key"  # Hardcode or load from env var
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

app = Flask(__name__, static_folder="static", static_url_path="")

# Global references (loaded once and cached)
DATAFRAME = None           # The main eBay DataFrame
EMBEDDINGS = None          # Numpy array of sentence embeddings
CLUSTER_LABELS = None      # HDBSCAN cluster labels
FAISS_INDEX = None         # FAISS index for fast similarity search
SENTENCE_MODEL = None      # SentenceTransformer model
CACHE = {}                 # Simple in-memory cache for repeated queries

# 1) Serve index.html
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# 2) Helper: Load eBay Data
def load_ebay_data():
    """
    Loads your CSV containing eBay product data, including columns like
    'Title', 'Price', 'Main Image', etc.
    """
    csv_path = r"your file path"
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None

# 3) Helper: Prepare Embeddings + Clustering + FAISS
def prepare_semantic_data(df):
    """
    - Tokenize titles (optional).
    - Generate sentence embeddings using SentenceTransformer.
    - Cluster embeddings using HDBSCAN.
    - Build a FAISS index for fast similarity searching.
    """
    # (A) Tokenize if needed
    df["Tokenized Title"] = df["Title"].astype(str).apply(lambda x: word_tokenize(x.lower()))

    # (B) Generate embeddings
    global SENTENCE_MODEL
    if SENTENCE_MODEL is None:
        # Load a SentenceTransformer model, e.g. 'all-MiniLM-L6-v2'
        SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    titles = df["Title"].astype(str).tolist()
    embeddings = SENTENCE_MODEL.encode(titles, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")  # FAISS requires float32

    # (C) Cluster using HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    df["Cluster"] = labels

    # (D) Build a FAISS index
    dim = embeddings.shape[1]  # dimension of embeddings
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)  # add all vectors

    return df, embeddings, labels, faiss_index

# 4) Helper: Chat with OpenAI
def chat_with_openai(messages):
    """
    Sends multi-turn conversation to OpenAI GPT (gpt-3.5-turbo).
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }
    try:
        resp = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.json().get("choices", [{}])[0].get("message", {}).get("content", 
                "Sorry, I couldn't process your request.")
        else:
            return "Sorry, I couldn't process your request."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with OpenAI API: {e}"

# 5) Helper: Perform Semantic Search with FAISS
def semantic_search_faiss(query, max_price=None):
    """
    Given a user query, compute its embedding, do a nearest neighbor search in FAISS,
    then filter by cluster or price if needed. Return top results.
    """
    global DATAFRAME, EMBEDDINGS, CLUSTER_LABELS, FAISS_INDEX, SENTENCE_MODEL

    # Basic caching to speed up repeated queries
    cache_key = (query, max_price)
    if cache_key in CACHE:
        return CACHE[cache_key]

    # 1) Compute embedding for the query
    query_embedding = SENTENCE_MODEL.encode([query], show_progress_bar=False).astype("float32")

    # 2) Use FAISS to find nearest neighbors
    k = 20  # retrieve top 20 neighbors, then refine
    distances, indices = FAISS_INDEX.search(query_embedding, k)
    # 'indices' is shape (1, k) with row indices of nearest neighbors
    # 'distances' is shape (1, k) with L2 distances

    # 3) Build a subset DataFrame of these neighbors
    neighbor_rows = DATAFRAME.iloc[indices[0]].copy()
    # Convert L2 distances to a similarity-like measure if you want
    neighbor_rows["Distance"] = distances[0]

    # 4) Filter by max_price if provided
    if max_price is not None:
        neighbor_rows = neighbor_rows[neighbor_rows["Price"] <= max_price]

    # 5) Sort by ascending distance (lowest = best match)
    neighbor_rows = neighbor_rows.sort_values("Distance", ascending=True)

    # 6) Return top 5
    results = neighbor_rows.head(5)

    # Cache it
    CACHE[cache_key] = results
    return results

# 6) Endpoint: Chat
@app.route("/chat", methods=["POST"])
def chat():
    """
    Expects JSON: { 'message': 'User message', 'chat_history': [ ... ] }
    """
    data = request.get_json()
    user_message = data.get("message", "")
    chat_history = data.get("chat_history", [])

    # Insert system message if not present
    system_message = {
        "role": "system",
        "content": (
            "You are an eBay shopping assistant specializing in shoes. "
            "Your expertise is in finding, comparing, and providing detailed information "
            "about shoes available on eBay. You respond only to shoe-related questions "
            "and avoid discussing products outside of footwear. Your tone is friendly, "
            "informative, and helpful. Use the content provided to guide users in "
            "making informed shoe purchases on eBay."
        )
    }
    if not chat_history or chat_history[0].get("role") != "system":
        chat_history.insert(0, system_message)

    # Append user message
    chat_history.append({"role": "user", "content": user_message})

    # Get GPT response
    response_text = chat_with_openai(chat_history)

    # Append GPT response
    chat_history.append({"role": "assistant", "content": response_text})

    return jsonify({"response": response_text, "chat_history": chat_history})

# 7) Endpoint: Search
@app.route("/search", methods=["POST"])
def search():
    """
    Expects JSON: { 'query': 'product query', 'max_price': optional }
    Returns top results from FAISS-based semantic search.
    """
    data = request.get_json()
    query = data.get("query", "")
    max_price = data.get("max_price")

    if DATAFRAME is None:
        return jsonify({"error": "Data not loaded"}), 500

    results_df = semantic_search_faiss(query, max_price)

    if results_df is None or results_df.empty:
        return jsonify({"results": []})

    # Convert to dict
    results = results_df.to_dict(orient="records")
    return jsonify({"results": results})

# 8) On Startup: Load Data, Build Embeddings, Clusters, and FAISS Index
@app.before_first_request
def load_data_and_prepare():
    global DATAFRAME, EMBEDDINGS, CLUSTER_LABELS, FAISS_INDEX
    df = load_ebay_data()
    if df is None:
        print("Could not load data. Please ensure your CSV file is present.")
        return

    # Precompute embeddings, clusters, FAISS index
    df, embeddings, labels, faiss_index = prepare_semantic_data(df)

    # Store globally
    DATAFRAME = df
    EMBEDDINGS = embeddings
    CLUSTER_LABELS = labels
    FAISS_INDEX = faiss_index

if __name__ == "__main__":
    app.run(debug=True)
