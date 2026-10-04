import json
import os
from pathlib import Path
from langchain_text_splitters import (  # Various splitting strategies
    RecursiveCharacterTextSplitter,  # Best general-purpose splitter
    CharacterTextSplitter,  # Simple split by character count
    MarkdownHeaderTextSplitter,  # Splits based on markdown headers
    HTMLHeaderTextSplitter  # Splits based on HTML tags
)
# AI-powered semantic chunking
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma  # Vector database
from langchain_openai import OpenAIEmbeddings  # OpenAI embedding function
from langchain_core.documents import Document  # Document abstraction
from dotenv import load_dotenv


# Load API keys from .env file (never hardcode API keys!)
load_dotenv()

# Load our support ticket dataset
# this is short document, in real scenarios, we need to chunk PDFs, articles, manuals (much longer!)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "mock_tickets.json"

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    tickets = json.load(f)
print(f"\nLoaded {len(tickets)} support tickets")

# First, convert our ticket data into LangChain Document objects
documents = []
for ticket in tickets:
    # Combine all ticket fields into a single text block
    # TIP: Include all relevant context that helps understand the document
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()

    # Create Document object with metadata. Metadata is CRUCIAL - it enables filtering later!
    # Example: "Find similar tickets, but only in the 'Authentication' category"
    doc = Document(
        page_content=full_text,  # The actual text content
        metadata={
            'ticket_id': ticket['ticket_id'],   # For identifying results
            'category': ticket['category'],      # For category filtering
            'priority': ticket['priority']       # For priority filtering
        }
    )
    documents.append(doc)

print(f"Created {len(documents)} documents")
print(f"\nSample document length: {len(documents[0].page_content)} characters")

# =============================================================================
# Fixed-Size Chunking
# =============================================================================
print("\n--------------- Strategy: Fixed-Size Chunking ------------------------")
fixed_splitter = CharacterTextSplitter(
    chunk_size=200,      # Maximum characters per chunk
    chunk_overlap=20,    # Characters to repeat between chunks (10% overlap)
    separator="\n"       # Prefer splitting on newlines when possible
)
fixed_chunks = fixed_splitter.split_documents(documents)

print(f"  Created {len(fixed_chunks)} chunks")
print("  Chunk size: 200 chars, Overlap: 20 chars")
print(f"  Sample chunk: {fixed_chunks[0].page_content[:100]}...")

# =============================================================================
# Recursive Character Splitting (RECOMMENDED DEFAULT)
# =============================================================================
print("\n--------------- Strategy: Recursive Chunking ------------------------")
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Max characters per chunk
    chunk_overlap=50,    # 50 char overlap (~17%)
    # Separators tried in ORDER - most specific first!
    separators=[
        "\n\n",  # 1st: Paragraph breaks (best split point)
        "\n",    # 2nd: Line breaks
        ". ",    # 3rd: Sentence boundaries
        " "      # 4th: Word boundaries (last resort for text)
    ]
)
recursive_chunks = recursive_splitter.split_documents(documents)

print(f"  Created {len(recursive_chunks)} chunks")
print("   Tries to split on paragraph/sentence boundaries")
print(f"  Sample chunk: {recursive_chunks[0].page_content[:100]}...")

# =============================================================================
# Semantic Chunking (Embedding-Based)
# =============================================================================
print("\n--------------- Strategy: Semantic Chunking -------------------------")

# Note: Semantic chunking uses embeddings to find natural break points
# Initialize OpenAI embeddings for semantic chunker
# IMPORTANT: This costs money! Each sentence needs an embedding API call
embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

# Demo with a paragraph that has CLEAR topic shifts
# Using completely unrelated domains for clearer separation
DEMO_TEXT = """
The Mars rover collected soil samples from the Jezero crater last week. Scientists believe these rocks may contain signs of ancient microbial life. NASA plans to retrieve these samples in a future mission. The discovery could reshape our understanding of life in the solar system.

Grandma's apple pie recipe starts with peeling six large Granny Smith apples. Mix flour, sugar, and cinnamon for the filling. Roll the dough thin and crimp the edges carefully. Bake at 375 degrees for 45 minutes until golden brown.

The defendant was charged with breach of contract under Section 12. The plaintiff seeks damages of fifty thousand dollars plus legal fees. Both parties agreed to mediation before proceeding to trial. The judge scheduled the preliminary hearing for next month.
"""

print("\n   Demo Text (3 distinct topics):")
print("  Topic 1: Space exploration (sentences 1-4)")
print("  Topic 2: Cooking recipe (sentences 5-8)")
print("  Topic 3: Legal case (sentences 9-12)")

semantic_splitter = SemanticChunker(
    embeddings=embeddings_model,
    # How to detect "topic change":
    # - "percentile": Split wheres similarity is in bottom X percentile
    # - "standard_deviation": Split where similarity is X std devs below mean
    # - "interquartile": Split where similarity is below Q1 - 1.5*IQR (outlier detection)
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.0  # Split when similarity drops 1 std dev below mean
)

demo_doc = Document(page_content=DEMO_TEXT.strip())
semantic_chunks = semantic_splitter.split_documents([demo_doc])
print(f"\n Created {len(semantic_chunks)} chunks")
print("  Note: Semantic chunking results vary based on embedding model and threshold settings")

# Show each semantic chunk
print("\n   Resulting Semantic Chunks:")
print("  " + "-"*70)
for i, chunk in enumerate(semantic_chunks):
    print(f"\n  Chunk {i+1} ({len(chunk.page_content)} chars):")
    print("  " + "~"*60)
    # Show full content for clarity
    for line in chunk.page_content.strip().split('\n'):
        if line.strip():
            print(f"    {line.strip()}")
    print("  " + "~"*60)

print("\n  The chunker attempts to detect topic shifts between space → cooking → legal")
print("  Adjust breakpoint_threshold_amount (lower = more sensitive) if results vary.")

# =============================================================================
# Markdown Structure-Aware Splitting
# =============================================================================
print("\n--------- Strategy: Markdown Structure-Aware Chunking ----------------")
MD_DOC = """
# Database Troubleshooting Guide

## Connection Issues

### Timeout Errors
If you encounter timeout errors, check the connection string and ensure the database server is reachable.
Increase the connection timeout value in your configuration.

### Authentication Failures
Verify your credentials are correct. Check for expired passwords or locked accounts.
Ensure the user has proper permissions on the database.

## Performance Problems

### Slow Queries
Analyze query execution plans using EXPLAIN.
Consider adding indexes on frequently queried columns.
Review and optimize JOIN operations.

### High CPU Usage
Monitor long-running queries.
Check for missing indexes causing table scans.
"""

# Define which headers to split on
# Format: (header_marker, metadata_key)
headers_to_split_on = [
    ("#", "Header 1"),    # H1 tags
    ("##", "Header 2"),   # H2 tags
    ("###", "Header 3"),  # H3 tags
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    # Keep headers in the chunk content (usually want True)
    strip_headers=False
)
md_chunks = markdown_splitter.split_text(MD_DOC)

print(f" Created {len(md_chunks)} chunks from markdown")
print("  Preserves document structure and header context")
if md_chunks:
    print("  Sample chunk with metadata:")
    print(f"    Content: {md_chunks[0].page_content[:80]}...")
    print(f"    Metadata: {md_chunks[0].metadata}")  # Shows header hierarchy!
# =============================================================================
# STRATEGY : HTML Structure-Aware Splitting
# =============================================================================
print("\n--------- Strategy: HTML Structure-Aware Chunking ----------------")
# Sample HTML documentation (simulating a scraped help page)
HTML_DOC = """
<!DOCTYPE html>
<html>
<body>
    <h1>Email Configuration Guide</h1>
    
    <h2>SMTP Settings</h2>
    <p>Configure your SMTP server settings in the admin panel. Use port 587 for TLS or port 465 for SSL.</p>
    
    <h3>Common SMTP Servers</h3>
    <p>Gmail: smtp.gmail.com, Outlook: smtp.office365.com, Yahoo: smtp.mail.yahoo.com</p>
    
    <h2>IMAP Configuration</h2>
    <p>Set up IMAP to sync your emails across devices. Use port 993 for secure connections.</p>
    
    <h3>Folder Mapping</h3>
    <p>Map your email folders to the appropriate IMAP folders for proper synchronization.</p>
</body>
</html>
"""

# Map HTML tags to metadata keys
headers_to_split_on_html = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on_html
)
html_chunks = html_splitter.split_text(HTML_DOC)

print(f"  Created {len(html_chunks)} chunks from HTML")
print("  Respects HTML semantic structure")
if html_chunks:
    print("   Sample chunk with metadata:")
    print(f"    Content: {html_chunks[0].page_content[:80]}...")
    print(f"    Metadata: {html_chunks[0].metadata}")

# ============================================================================
# Chroma Vector Store
# ============================================================================
embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

QUERY = "Authentication problems after password reset"
print("\nBuilding Chroma vector store...")

# Clean up any existing collection to avoid duplicates on re-run
existing_store = Chroma(
    collection_name="support_tickets",
    persist_directory="./chroma_db"
)
existing_store.delete_collection()

# from_documents() handles everything:
#   1. Extracts text from each Document
#   2. Generates embeddings via the embedding model
#   3. Stores vectors + metadata + original text
#   4. Persists to disk (if persist_directory specified)
chroma_store = Chroma.from_documents(
    documents=documents,              # Our LangChain Document objects
    embedding=embeddings_model,       # OpenAI embeddings
    collection_name="support_tickets",  # Like a "table" in a database
    persist_directory="./chroma_db"   # Save to disk for persistence
)
print(" Chroma store created and persisted")

# Basic Similarity Search
print(f"\nSearching in Chroma: '{QUERY}'")
chroma_results = chroma_store.similarity_search(QUERY, k=3)

print(f"\nTop {len(chroma_results)} results:")
for i, doc in enumerate(chroma_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")

# MMR Search (Maximal Marginal Relevance)
print("\n--- Using MMR for Diverse Results ---")
mmr_results = chroma_store.max_marginal_relevance_search(QUERY, k=3)

print("\nMMR Results (more diverse):")
for i, doc in enumerate(mmr_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(
        f"Title: {tickets[int(doc.metadata['ticket_id'].split('-')[1]) - 1]['title']}")

# ============================================================================
# Metadata Filtering
# ============================================================================

# Example 1: Filter by category
print("\nSearching only in 'Authentication' category:")
filtered_results = chroma_store.similarity_search(
    QUERY,
    k=3,
    filter={"category": "Authentication"}  # Only match this category
)

print(f"\nFiltered results ({len(filtered_results)}):")
for i, doc in enumerate(filtered_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")
    print(f"Content: {doc.page_content[:100]}...")

# Example 2: Filter by priority
print("\n\nSearching only 'High' priority tickets:")
high_priority_results = chroma_store.similarity_search(
    "Database performance issues",
    k=3,
    filter={"priority": "High"}  # Only high priority
)

print(f"\nHigh priority results ({len(high_priority_results)}):")
for i, doc in enumerate(high_priority_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Priority: {doc.metadata['priority']}")

# ============================================================================
# Comparing Chunking Strategies
# ============================================================================

# Build stores with different chunking
print("\nBuilding vector stores with different chunking strategies...")

# Store 1: Whole documents (no chunking)
store_whole = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name="whole_docs"
)

# Store 2: Fixed-size chunks (may split mid-sentence)
store_fixed = Chroma.from_documents(
    documents=fixed_chunks,
    embedding=embeddings_model,
    collection_name="fixed_chunks"
)

# Store 3: Recursive chunks (splits at natural boundaries)
store_recursive = Chroma.from_documents(
    documents=recursive_chunks,
    embedding=embeddings_model,
    collection_name="recursive_chunks"
)

TEST_QUERY = "Database connection failures"
print(f"\nTest query: '{TEST_QUERY}'")

# Compare results from each strategy
stores = [
    ("Whole Documents", store_whole),
    ("Fixed Chunks", store_fixed),
    ("Recursive Chunks", store_recursive)
]

for name, store in stores:
    results = store.similarity_search(TEST_QUERY, k=1)
    print(f"\n{name}:")
    if results:
        print(f"  Top result: {results[0].page_content[:100]}...")
        print(f"  Length: {len(results[0].page_content)} chars")
