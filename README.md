# Observable RAG System with FastAPI, Qdrant, and Phoenix

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)
![Phoenix](https://img.shields.io/badge/Phoenix-Observability-orange)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Tracing-purple)

An end-to-end Retrieval-Augmented Generation (RAG) backend system built using FastAPI, Qdrant, Sentence Transformers, Gemini, and OpenTelemetry-based observability with Phoenix.

This project focuses on building a production-style AI backend architecture with:

* document ingestion pipelines
* semantic retrieval
* vector search
* prompt orchestration
* telemetry and tracing for debugging RAG workflows

---

# Features

* PDF document ingestion pipeline
* Semantic chunking and embedding generation
* Vector similarity search using Qdrant
* Retrieval-Augmented Generation (RAG)
* Gemini-powered response generation
* OpenTelemetry tracing
* Phoenix observability integration
* Modular backend architecture
* Environment-based configuration management
* Test scripts for ingestion, retrieval, and generation

---

# Architecture

## Query Flow

```text
User Query
     ↓
FastAPI Endpoint
     ↓
RAG Retrieval Pipeline
     ↓
Qdrant Vector Search
     ↓
Relevant Context Retrieval
     ↓
Prompt Construction
     ↓
Gemini LLM Generation
     ↓
Final Response
```

## Document Ingestion Flow

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embedding Generation
    ↓
Qdrant Vector Storage
```

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

## 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=rag_documents

GEMINI_API_KEY=your_api_key
```

## 5. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 6. Start Phoenix

```bash
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix
```

Phoenix UI:

```text
http://localhost:6006
```

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

## Ingest PDF

```http
POST /ingest
```

Uploads and processes PDF documents into the vector database.

## Chat with RAG

```http
POST /chat
```

Retrieves relevant chunks and generates contextual responses using Gemini.

---

# Sample API Requests

## Ingest PDF

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
-F "file=@sample.pdf"
```

## Chat Endpoint

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
-H "Content-Type: application/json" \
-d '{
  "query": "What is technical analysis?"
}'
```

---

# Postman Collection

A sample Postman collection is included for testing the API endpoints:

```text
postman/Rag_system_apis.postman_collection.json
```

The collection includes:

* PDF ingestion requests
* RAG chat requests

Import the collection into Postman and start testing immediately after running the backend server.

---

# Telemetry & Observability

This project integrates OpenTelemetry and Phoenix to trace:

* vector retrieval
* prompt construction
* LLM generation
* retrieved context
* final responses

The observability layer helps debug:

* poor retrieval quality
* noisy chunking
* hallucinations
* prompt construction issues

---

# Current Limitations

* Basic character-based chunking
* PDF-only ingestion
* No reranking layer
* No hybrid search
* Limited metadata filtering

---

# Future Improvements

* Semantic chunking
* Metadata-aware retrieval
* Hybrid search (BM25 + vector search)
* Reranking pipelines
* Source citations
* Streaming responses
* Docker Compose setup
* Kubernetes deployment
* Evaluation pipelines
* Multi-modal ingestion

---

# Learning Outcomes

This project was built to deeply understand:

* RAG system architecture
* vector databases
* embedding pipelines
* observability in AI systems
* OpenTelemetry tracing
* production-style backend engineering patterns
* retrieval quality debugging
