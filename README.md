# rag-support-desk: Rag Support Desk Ticket Retrieval & Troubleshooting Assistant

A RAG system built with OpenAI embeddings, LangChain, and LlamaIndex. This workshop guides you through constructing an incident response assistant that retrieves technical ticket context while enforcing anti-hallucination safeguards.

---

## Key Learning Objectives

- **Embeddings & Chunking:** Generate OpenAI embeddings and implement optimal chunking strategies for incident ticket datasets.
- **Indexing Strategies:** Implement and compare 5 distinct LlamaIndex retrieval strategies.
- **Pipeline & Safeguards:** Build end-to-end LangChain RAG pipelines with strict anti-hallucination checks.
- **Evaluation:** Measure system performance using two-layer retrieval and generation metrics.
- **Agentic RAG:** Develop multi-step reasoning capabilities for automated troubleshooting.

---

## Quick Start

### Prerequisites

- **Python 3.12** _(Python 3.13+ is unsupported due to `chromadb depends on Pydantic V1 internals that were removed in Python 3.13+.)_
- An active **OpenAI API Key**

### Setup Instructions

#### 1. Clone & Set Up Virtual Environment

**macOS / Linux:**

```bash
git clone <repo-url>
cd rag-support-desk
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
git clone <your-repo-url>
cd rag-support-desk
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Environment Configuration

Copy the template configuration file:

```bash
# macOS/Linux:
cp .env.example .env

# Windows (PowerShell):
Copy-Item .env.example .env
```

Set your API credentials in `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

#### 4. Run Smoke Test

Verify your setup by running the first module:

```bash
cd modules/1_embeddings
python embeddings.py
```
