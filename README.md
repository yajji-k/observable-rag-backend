# Observable RAG System with FastAPI, Qdrant, and Phoenix

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)
![Phoenix](https://img.shields.io/badge/Phoenix-Observability-orange)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Tracing-purple)

An end-to-end Retrieval-Augmented Generation (RAG) backend system built using FastAPI, Qdrant, Sentence Transformers, Gemini, and OpenTelemetry-based observability with Phoenix.

The project focuses on understanding production-style AI backend architecture through:

* ingestion engineering
* retrieval engineering
* observability-driven debugging
* vector search systems
* prompt orchestration
* semantic retrieval pipelines

---

# Features

* PDF document ingestion pipeline
* Strategy-based chunking architecture
* Character chunking
* Recursive chunking
* Token chunking
* Semantic chunking
* Dynamic chunking strategy selection
* Chunking strategy discovery endpoint
* Text cleaning and normalization pipeline
* Vector similarity search using Qdrant
* Retrieval-Augmented Generation (RAG)
* Gemini-powered response generation
* OpenTelemetry tracing
* Phoenix observability integration
* Retrieval context observability
* Strategy-specific vector collections
* Modular backend architecture
* Environment-based configuration management
* Postman collection for API testing
* Test scripts for ingestion, retrieval, and generation

---

# Architecture

## Query Flow

```text
User Query
     ↓
FastAPI Endpoint
     ↓
Query Embedding Generation
     ↓
Qdrant Vector Search
     ↓
Metadata-Aware Retrieval
     ↓
Retrieval Score Analytics
     ↓
Context Construction
     ↓
Prompt Construction
     ↓
Gemini LLM Generation
     ↓
Final Response
```

---

## Document Ingestion Flow

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Chunking Strategy Selection
    ↓
Embedding Generation
    ↓
Qdrant Vector Storage
```

---

# Chunking Strategies

The ingestion pipeline uses a pluggable Strategy Pattern architecture, allowing new chunking methods to be added with minimal changes to the ingestion pipeline.

Supported strategies:

## Character Chunking

Fixed-size chunk slicing with overlap.

Useful for:

* baseline retrieval
* simple ingestion pipelines
* fast experimentation

---

## Recursive Chunking

Implemented using:

```text
RecursiveCharacterTextSplitter
```

Preserves:

* sentence continuity
* paragraph structure
* semantic boundaries

Useful for:

* structured documents
* general-purpose retrieval
* context preservation

---

## Token Chunking

Splits documents based on token counts rather than character counts.

Benefits:

* chunk sizes align with LLM tokenization
* more predictable embedding sizes
* improved context window management

Useful for:

* embedding experiments
* token-aware retrieval pipelines
* LLM context optimization

---

## Semantic Chunking

Groups neighboring sentences based on embedding similarity.

Workflow:

```text
Document
    ↓
Sentence Splitting
    ↓
Sentence Embeddings
    ↓
Cosine Similarity
    ↓
Semantic Grouping
```

Benefits:

* topic-aware chunk boundaries
* improved semantic coherence
* retrieval-focused chunk construction

Useful for:

* retrieval experimentation
* semantic search optimization
* chunk quality evaluation

---

# Text Cleaning Pipeline

Before chunking, extracted PDF text passes through a preprocessing layer to reduce retrieval noise.

Current preprocessing includes:

* whitespace normalization
* newline cleanup
* standalone page number removal
* PDF ligature normalization

This improved:

* embedding quality
* retrieval readability
* prompt quality

---

# Metadata-Aware Retrieval

Each chunk stored in Qdrant contains retrieval metadata:

```json
{
  "text": "...",
  "source_file": "TA_wrkbk.pdf",
  "chunk_id": 348,
  "chunk_strategy": "recursive",
  "domain": "general"
}
```

Metadata enables:

* source tracking
* chunk traceability
* retrieval debugging
* chunking strategy comparisons
* observability-driven retrieval analysis

During retrieval, chunk metadata is surfaced in Phoenix traces to help inspect retrieval quality and understand why specific chunks were selected.

---

# Retrieval Score Analytics

The retrieval pipeline captures score analytics for every query and exposes them through Phoenix traces.

Tracked metrics:

* retrieval.max_score
* retrieval.min_score
* retrieval.avg_score
* retrieval.score_range

Example telemetry:

```json
{
  "retrieval": {
    "max_score": 0.8740,
    "avg_score": 0.8320,
    "min_score": 0.8071,
    "score_range": 0.0669
  }
}
```

These metrics help evaluate:

* retrieval quality
* chunking strategies
* embedding model effectiveness
* retrieval consistency
* future similarity threshold selection

The goal is to measure retrieval quality before introducing threshold-based filtering or more advanced retrieval techniques.

---

# Tech Stack

## Backend

* FastAPI
* Python

## Vector Database

* Qdrant

## Embeddings

* Sentence Transformers
* BAAI/bge-small-en-v1.5

## LLM

* Google Gemini

## Observability

* OpenTelemetry
* Arize Phoenix

## Document Processing

* PyPDF

---

# Project Structure

```text
app/
├── api/
├── generation/
├── ingestion/
├── rag/
├── retrieval/
├── telemetry.py
├── config.py
└── main.py

tests/
├── test_generation.py
├── test_ingestion.py
├── test_loader.py
└── test_retrieval.py

postman/
└── Rag_system_apis.postman_collection.json
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <your_repo_url>
cd observable-rag-backend
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333

GEMINI_API_KEY=your_api_key
```

---

## 5. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Qdrant Dashboard:

```text
http://localhost:6333/dashboard
```

---

## 6. Start Phoenix

```bash
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix
```

Phoenix UI:

```text
http://localhost:6006
```

---

## 7. Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

FastAPI Docs:

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Ingest Document

```http
POST /ingest
```

Uploads and processes PDF documents into the vector database.

Supports configurable chunking strategies.

---

## Chat with RAG

```http
POST /chat
```

Retrieves relevant chunks and generates contextual responses using Gemini.

---

# Sample API Requests

## Ingest PDF with Recursive Chunking

```bash
curl --location 'http://127.0.0.1:8000/ingest' \
--form 'file=@"/path/to/document.pdf"' \
--form 'chunk_strategy=recursive'
```

---

## Chat Endpoint

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
-H "Content-Type: application/json" \
-d '{
  "query": "What is technical analysis?",
  "chunk_strategy": "recursive"
}'
```

---

# Supported Chunking Strategies

| Strategy  | Description                       |
| --------- | --------------------------------- |
| character | Fixed-size chunk slicing          |
| recursive | Recursive semantic-aware chunking |

---

# Postman Collection

A Postman collection is included for testing the APIs:

```text
postman/Rag_system_apis.postman_collection.json
```

The collection contains:

* ingestion requests
* RAG chat requests
* multipart form-data examples

---

## Important Postman Note

If multipart form-data parsing fails in Postman:

* generate the cURL command from Postman
* run the generated cURL directly in the terminal

The backend ingestion endpoint works correctly with terminal cURL execution.

---

# Observability & Tracing

This project integrates OpenTelemetry and Phoenix to trace:
This project integrates OpenTelemetry and Phoenix to trace:

* user queries
* vector retrieval
* retrieval scores
* retrieval score analytics
* source files
* chunk IDs
* retrieved context
* prompt construction
* selected vector collection
* LLM generation
* final responses

Telemetry is heavily used to:

* inspect chunk quality
* debug noisy retrieval
* compare chunking strategies
* identify PDF extraction artifacts
* improve ingestion quality iteratively

---

# Current Limitations

* PDF-only ingestion
* No reranking layer
* No hybrid search
* No retrieval score thresholding
* No source citations
* Basic retrieval evaluation only

---

# Future Improvements

* Semantic chunking
* Similarity threshold filtering
* Hybrid search (BM25 + vector search)
* Reranking pipelines
* Source citations
* Streaming responses
* Docker Compose setup
* Kubernetes deployment
* Evaluation pipelines
* Multi-modal ingestion
* Hierarchical retrieval
* Agent memory integration

---

# Learning Outcomes

This project was built to deeply understand:

* RAG system architecture
* vector databases
* ingestion engineering
* retrieval engineering
* embedding pipelines
* chunking tradeoffs
* AI observability
* OpenTelemetry tracing
* prompt grounding strategies
* production-style backend engineering patterns
* retrieval quality debugging

A major focus of the project is understanding how ingestion quality directly impacts:

* embeddings
* retrieval quality
* prompt quality
* final generation quality

rather than simply assembling AI frameworks together.

---

# Engineering Workflow

The project follows an observability-first retrieval engineering workflow:

```text
Build Baseline Pipeline
        ↓
Observe Retrieval Failures
        ↓
Inspect Telemetry Traces
        ↓
Improve Ingestion/Retrieval
        ↓
Re-evaluate Response Quality
```

This iterative workflow mirrors how real-world RAG systems are optimized in production environments.
