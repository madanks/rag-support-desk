import json
import os
from pathlib import Path
# OpenAI API client for generating embeddings
from openai import OpenAI
# For numerical operations on embedding vectors
import numpy as np
# Measure similarity between vectors
from sklearn.metrics.pairwise import cosine_similarity
# Load environment variables from .env file
from dotenv import load_dotenv

# =============================================================================
# SETUP: Load Environment Variables
# =============================================================================
# NEVER hardcode API keys in your code
# Store them in a .env file and load with python-dotenv
# .env file should contain:OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL
# =============================================================================
load_dotenv()

# =============================================================================
# INITIALIZE OPENAI CLIENT
# =============================================================================
print("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# =============================================================================
# EMBEDDING MODEL SELECTION
# =============================================================================
embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
print(f"Using OpenAI model: {embedding_model}")

# =============================================================================
# LOAD DATA: For now Mock Support Tickets
# =============================================================================
print("\nLoading support tickets...")
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "mock_tickets.json"

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    tickets = json.load(f)
print(f"Loaded {len(tickets)} support tickets")

# -----------------------------------------------------------------------------
# Prepare text for embedding
# -----------------------------------------------------------------------------
# TIP: Combine relevant fields for richer context
# The more context, the better the embedding captures the meaning
# -----------------------------------------------------------------------------
ticket_texts = [
    f"{ticket['title']}. {ticket['description']}"
    for ticket in tickets
]

# -----------------------------------------------------------------------------
# Generate embeddings via OpenAI API
# COST: ~$0.02 per 1M tokens for text-embedding-3-small
# -----------------------------------------------------------------------------
print("\nGenerating embeddings for all tickets...")
response = client.embeddings.create(input=ticket_texts, model=embedding_model)

# Convert API response to NumPy array for mathematical operations
print("\nGenerating embeddings for all tickets...")
response = client.embeddings.create(input=ticket_texts, model=embedding_model)
embeddings = np.array([data.embedding for data in response.data])
print(f"Generated embeddings with shape: {embeddings.shape}")
print(f"({len(tickets)} dimensions)")

# -----------------------------------------------------------------------------
# Generate embedding for the query
# -----------------------------------------------------------------------------

# Search query - user input query
QUERY = "Users can't login after changing password"
print(f"\nSearch Query: '{QUERY}'")

query_response = client.embeddings.create(input=[QUERY], model=embedding_model)
query_embedding = np.array([query_response.data[0].embedding])
print(f"Query embedding shape: {query_embedding.shape}")  # (1, 1536)

# -----------------------------------------------------------------------------
# Compute cosine similarity between query and ALL tickets
# There is only one query row, so [0] extracts that row
# -----------------------------------------------------------------------------
similarities = cosine_similarity(query_embedding, embeddings)[0]
print(f"\nComputed similarity scores for {len(similarities)} tickets")
print(
    f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")

# ============================================================================
# Retrieve Most Similar Tickets : semantic search
# ============================================================================

# Get top 5 most similar tickets
TOP_K = 5

# np.argsort() returns indices that would sort the array (ascending)
# [::-1] reverses to get descending order (highest similarity first)
# [:top_k] takes only the top K results
top_indices = np.argsort(similarities)[::-1][:TOP_K]
print(f"\nTop {TOP_K} most similar tickets to query: '{QUERY}'")

for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]

    print(f"\n#{rank} - Similarity: {score:.4f}")
    print(f"Ticket ID: {ticket['ticket_id']}")
    print(f"Title: {ticket['title']}")
    print(f"Category: {ticket['category']} | Priority: {ticket['priority']}")
    print(f"Description: {ticket['description'][:150]}...")

# ============================================================================
# Experiment with Different Queries
# ============================================================================

TEST_QUERIES = [
    "Database is timing out",
    "Payment not working for foreign customers",
    "App crashes on iPhone",
    "Emails are not being sent"
]
print("\nTesting semantic search with different queries one by one")
for test_query in TEST_QUERIES:
    # Generate query embedding
    query_resp = client.embeddings.create(
        input=[test_query], model=embedding_model)
    query_emb = np.array([query_resp.data[0].embedding])

    # Compare to all tickets
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argmax(sims)

    print(f"\nQuery: '{test_query}'")
    print(f"  → Best match: {tickets[top_idx]['title']}")
    print(f"  → Similarity: {sims[top_idx]:.4f}")
