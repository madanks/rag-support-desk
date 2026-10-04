import json
import os
from pathlib import Path
from dotenv import load_dotenv

# LlamaIndex core components
from llama_index.core import (
    VectorStoreIndex,    # Standard embedding-based index
    SummaryIndex,        # Full document storage, LLM-based relevance
    TreeIndex,           # Hierarchical summarization tree
    KeywordTableIndex,   # Inverted keyword index
    Document,            # Document wrapper with text + metadata
    Settings             # Global configuration
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Load Environment Variables
load_dotenv()

# Set longer timeout for httpx (used by OpenAI client)
# Some index types make MANY LLM calls and need more time
os.environ["HTTPX_TIMEOUT"] = "300"  # 5 minutes

# =============================================================================
# CONFIGURE LLAMAINDEX SETTINGS
# =============================================================================
# LlamaIndex uses a Settings singleton to configure:
#   - embed_model: Which embedding model to use
#   - llm: Which LLM to use for queries and index building
Settings.embed_model = OpenAIEmbedding(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=120,      # 2 min timeout for embedding calls
    max_retries=5     # Retry on failure
)
Settings.llm = OpenAI(
    model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=300,      # 5 min timeout (Tree/Keyword indexes are slow!)
    max_retries=5
)

# LOAD DATA
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "mock_tickets.json"

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    tickets = json.load(f)
print(f"\nLoaded {len(tickets)} support tickets")


# Convert tickets to LlamaIndex Documents
documents = []
for ticket in tickets:
    # Combine all fields into content (rich context for embedding)
    # IMPORTANT: Include ticket_id in text so keyword index can find it!
    content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""

    doc = Document(
        text=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'title': ticket['title']
        }
    )
    documents.append(doc)

print(f" Loaded {len(documents)} support tickets")

# Test query - we'll use this across all index types
QUERY = "How do I fix authentication issues after password reset?"
print(f"\nTest Query: '{QUERY}'")

# ============================================================================
# Vector Index (Flat Index)
# ============================================================================

# Build the Vector Index
vector_index = VectorStoreIndex.from_documents(documents)

# Create Query Engine
vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)

print(" Created vector index")
print(f"\nQuery: '{QUERY}'")
vector_response = vector_query_engine.query(QUERY)

print("\nVector Index Results:")
print(f"Answer: {vector_response.response}\n")
print("Source Documents:")
for i, node in enumerate(vector_response.source_nodes, 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    # Similarity score (higher = more similar)
    print(f"   Score: {node.score:.4f}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# Summary Index
# ============================================================================

# Build the Summary Index
summary_index = SummaryIndex.from_documents(documents)

# Query Engine with Tree Summarize
# response_mode="tree_summarize":
#   1. Collects all relevant documents
#   2. If too many, summarizes in groups
#   3. Combines group summaries into final answer
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize")

print(" Created summary index")
print(f"\nQuery: '{QUERY}'")

summary_response = summary_query_engine.query(QUERY)

print("\nSummary Index Results:")
print(f"Answer: {summary_response.response}\n")
print("Source Documents:")
for i, node in enumerate(summary_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# Tree Index (Hierarchical Retrieval)
# ============================================================================

# Use all documents (but warn about LLM costs)
tree_documents = documents
print(f"Building Tree Index with {len(tree_documents)} documents...")

# Build the Tree Index
tree_index = TreeIndex.from_documents(tree_documents)

# Create Query Engine
# child_branch_factor controls how many branches the LLM follows at each level:
#
#   =1 (greedy):  Fast but may miss info in other branches
#                  Good for focused single-topic queries
#   =2 (balanced): Explores top 2 branches per level — catches multi-topic
#                  queries like "auth AND billing issues" (recommended)
#   =N (all):     Explores everything — maximum recall but slow,
#                  approaches Summary Index behavior
tree_query_engine = tree_index.as_query_engine(child_branch_factor=2)

print(" Created tree index with hierarchical structure")
print(f"\nQuery: '{QUERY}'")

# Query execution (hierarchical traversal):
# 1. Start at root summary node
# 2. LLM scores each child: "Is this branch relevant to the query?"
# 3. Select top `child_branch_factor` branches (here: top 2)
# 4. Expand those branches → evaluate their children
# 5. Repeat until reaching leaf nodes (actual document content)
# 6. Collect all relevant leaves from explored paths
# 7. Synthesize final answer from collected leaves
tree_response = tree_query_engine.query(QUERY)

print("\nTree Index Results:")
print(f"Answer: {tree_response.response}\n")
print("Source Documents:")
for i, node in enumerate(tree_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# Keyword Table Index
# ============================================================================
keyword_documents = documents
print(f"Building Keyword Index with {len(keyword_documents)} documents...")

# Build the Keyword Table Index
# LlamaIndex uses LLM to extract keywords from each document
# Builds inverted index: keyword → [document IDs]
# Alternative: Use simple regex/rule-based extraction (faster, no LLM)

keyword_index = KeywordTableIndex.from_documents(keyword_documents)

# Show the extracted keyword table (inverted index)
keyword_table = keyword_index.index_struct.table
print(f"\n✓ Extracted {len(keyword_table)} unique keywords:")
for keyword, node_ids in sorted(keyword_table.items()):
    print(f"  '{keyword}' → {len(node_ids)} document(s)")

# Create query engine
keyword_query_engine = keyword_index.as_query_engine()

print("\n✓ Created keyword table index")
print(f"\nQuery: '{QUERY}'")

# Query process:
# 1. Extract keywords from query (via LLM)
# 2. Look up documents in inverted index
# 3. Return documents containing query keywords
# 4. Synthesize answer from matched documents
keyword_response = keyword_query_engine.query(QUERY)

print("\nKeyword Index Results:")
print(f"Answer: {keyword_response.response}\n")
print("Source Documents:")
for i, node in enumerate(keyword_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# Hybrid Retrieval
# ============================================================================

# Step 1: Retrieve from Vector Index (Semantic)
vector_nodes = vector_index.as_retriever(similarity_top_k=5).retrieve(QUERY)

# Step 2: Retrieve from Keyword Index (Exact)
keyword_nodes = keyword_index.as_retriever().retrieve(QUERY)

# Step 3: Fusion - Combine and Deduplicate
seen_ids = set()
hybrid_nodes = []

for node in vector_nodes + keyword_nodes:
    node_id = node.metadata.get('ticket_id', node.node_id)
    if node_id not in seen_ids:
        seen_ids.add(node_id)
        hybrid_nodes.append(node)

# Documents found by BOTH methods are likely most relevant!

print("\nHybrid Retrieval Results (Combined):")
for i, node in enumerate(hybrid_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    if hasattr(node, 'score') and node.score:
        print(f"   Score: {node.score:.4f}")
    print(f"   {node.text[:150]}...")
