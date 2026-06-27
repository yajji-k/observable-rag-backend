# Observable RAG Backend

### Production-Grade Retrieval-Augmented Generation with FastAPI, Qdrant, Gemini, Phoenix, and Cross-Encoder Reranking

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-red)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-Embeddings-blueviolet)
![Google Gemini](https://img.shields.io/badge/Google-Gemini-blue)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Tracing-purple)
![Phoenix](https://img.shields.io/badge/Arize-Phoenix-orange)

---

# Overview

Observable RAG Backend is a production-style Retrieval-Augmented Generation (RAG) system built to explore modern AI Engineering practices beyond basic document question answering.

Unlike traditional RAG applications that stop at vector search and LLM prompting, this project focuses on the complete retrieval lifecycle, including:

* Modular document ingestion
* Multiple chunking strategies
* Dense vector retrieval
* Cross Encoder reranking
* Retrieval benchmarking
* Retrieval quality evaluation
* OpenTelemetry instrumentation
* Phoenix observability
* Extensible architecture

The primary goal is to build a retrieval system that is measurable, observable, and continuously improvable rather than relying solely on prompt engineering.

---

# Why This Project?

Most introductory RAG projects follow a simple pipeline:

```text
PDF
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector Database
   ↓
LLM
```

While this works, production retrieval systems require much more.

This project extends the traditional RAG architecture with:

* Multiple chunking algorithms
* Retrieval benchmarking
* Retrieval quality metrics
* Cross Encoder reranking
* End-to-end tracing
* Retrieval analytics
* Modular architecture
* Configuration-driven retrieval

The focus is not only generating answers, but understanding **why** a document was retrieved, **how well** retrieval performs, and **how retrieval quality can be improved**.

---

# Features

## Document Ingestion

* PDF ingestion
* Folder-based ingestion
* Text extraction
* Text cleaning and normalization
* Metadata preservation
* Batch embedding generation
* Automatic Qdrant collection creation

---

## Chunking

Four independent chunking strategies are implemented.

* Character Chunking
* Recursive Chunking
* Token Chunking
* Semantic Chunking

Each strategy stores documents in an independent Qdrant collection, allowing retrieval performance to be benchmarked independently.

---

## Retrieval

* Dense vector retrieval
* Qdrant similarity search
* Configurable Top-K retrieval
* Strategy-specific collections
* Retrieval analytics
* Optional Cross Encoder reranking

---

## Reranking

Production-style reranking is implemented using:

```
BAAI/bge-reranker-base
```

through the Sentence Transformers CrossEncoder.

The reranking stage:

* retrieves Top N vector candidates
* evaluates each query-document pair
* computes semantic relevance scores
* reranks candidates
* returns only the Top K most relevant chunks

without modifying the existing retrieval pipeline.

---

## Retrieval Evaluation

The evaluation framework supports:

* Recall@K
* Precision@K
* Mean Reciprocal Rank (MRR)
* Normalized Discounted Cumulative Gain (NDCG)
* Retrieval latency
* Average retrieval score

allowing objective comparison between chunking strategies and reranking configurations.

---

## Observability

OpenTelemetry and Arize Phoenix are integrated throughout the application.

Tracing covers:

* Document ingestion
* Embedding generation
* Vector retrieval
* Cross Encoder reranking
* Prompt construction
* Gemini generation
* Retrieval evaluation
* Benchmark execution

making every stage of the RAG pipeline observable.

---

# High-Level Architecture

```text
                        +------------------+
                        |      PDF         |
                        +--------+---------+
                                 |
                                 v
                    +-------------------------+
                    |   Text Extraction       |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |    Text Cleaning        |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Chunking Strategy        |
                    |-------------------------|
                    | Character               |
                    | Recursive               |
                    | Token                   |
                    | Semantic                |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Embedding Generation    |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |        Qdrant           |
                    +------------+------------+
                                 |
                                 |
                    ==========================
                          Retrieval
                    ==========================
                                 |
                                 v
                         User Query
                                 |
                                 v
                     Query Embedding
                                 |
                                 v
                  Vector Search (Top N)
                                 |
                                 v
                Cross Encoder Reranker
                                 |
                                 v
                Top K Relevant Chunks
                                 |
                                 v
                  Prompt Construction
                                 |
                                 v
                        Google Gemini
                                 |
                                 v
                        Final Response
```

---

# Retrieval Pipeline

## Vector Retrieval

When reranking is disabled, retrieval follows a traditional dense retrieval pipeline.

```text
User Query
      ↓
Embedding Model
      ↓
Qdrant Vector Search
      ↓
Top K Chunks
      ↓
Context Builder
      ↓
Prompt Builder
      ↓
Gemini
```

This provides fast retrieval using embedding similarity.

---

## Retrieval with Cross Encoder Reranking

When reranking is enabled, the retrieval pipeline becomes:

```text
User Query
      ↓
Embedding Model
      ↓
Qdrant Vector Search (Top N Candidates)
      ↓
Cross Encoder Reranker
      ↓
Top K Reranked Chunks
      ↓
Context Builder
      ↓
Prompt Builder
      ↓
Gemini
```

The vector database is responsible for fast candidate generation.

The Cross Encoder then evaluates each query-document pair jointly, producing significantly better ranking quality than embedding similarity alone.

---

# Why Reranking Happens After Vector Search

Running a Cross Encoder across an entire document corpus is computationally expensive.

Instead, production systems first retrieve a small candidate set using vector search.

Example:

```text
Corpus

500,000 chunks

        ↓

Vector Search

Top 20

        ↓

Cross Encoder

Top 20 only

        ↓

Return Top 5
```

This architecture combines:

* Fast retrieval
* High semantic accuracy
* Predictable latency

and is the standard retrieval architecture used in modern production RAG systems.

---

# Reranking Architecture

The reranking layer follows the Strategy Pattern.

```text
BaseReranker
      │
      ▼
BGEReranker
```

Retrieval depends only on the abstraction.

```text
Retriever
      │
      ▼
RerankingService
      │
      ▼
RerankerRegistry
      │
      ▼
BaseReranker
      │
      ▼
BGEReranker
```

Future rerankers can be introduced without modifying the retrieval service.

Examples include:

* Cohere Rerank
* Jina AI Reranker
* Voyage AI Reranker
* Additional Cross Encoder models

---

# Reranker Configuration

```env
RERANKER_ENABLED=false

RERANKER_MODEL=bge

RERANKER_BGE_MODEL_NAME=BAAI/bge-reranker-base

RERANKER_CANDIDATE_COUNT=20

RERANKER_FINAL_TOP_K=5
```

When disabled, retrieval behaves exactly like a standard dense vector retrieval system.

When enabled:

* retrieve Top 20 candidates
* rerank using Cross Encoder
* return Top 5 results

without changing the public API.

---

# Document Ingestion Pipeline

Every uploaded PDF passes through the following pipeline.

```text
PDF Upload
      ↓
Text Extraction
      ↓
Cleaning & Normalization
      ↓
Chunking Strategy
      ↓
Embedding Generation
      ↓
Qdrant Storage
```

Each chunk is stored together with metadata.

Example:

```json
{
  "text": "...",
  "source_file": "technical_analysis.pdf",
  "document_name": "technical_analysis.pdf",
  "chunk_id": 42,
  "chunk_strategy": "semantic"
}
```

The metadata is later used for:

* retrieval evaluation
* benchmark execution
* Phoenix trace inspection
* debugging
* source attribution

---

# Chunking Strategies

The ingestion pipeline uses a Strategy Pattern allowing chunking algorithms to be swapped independently.

| Strategy  | Description                                  |
| --------- | -------------------------------------------- |
| Character | Fixed-size character chunks with overlap     |
| Recursive | Structure-aware recursive chunking           |
| Token     | Token-count based chunking                   |
| Semantic  | Embedding similarity based semantic grouping |

Each strategy writes to its own Qdrant collection, making benchmark comparisons deterministic and reproducible.

Available chunking strategies can be discovered through:

```http
GET /chunking/strategies
```

---

---

# Observability

Understanding retrieval behavior is just as important as generating accurate answers.

This project integrates **OpenTelemetry** with **Arize Phoenix** to provide end-to-end visibility into every stage of the RAG pipeline.

Rather than treating the system as a black box, every important operation is instrumented with traces, spans, attributes, and retrieval metadata.

---

## Traced Components

The following components are instrumented:

* PDF ingestion
* Embedding generation
* Vector retrieval
* Cross Encoder reranking
* Context construction
* Prompt generation
* Gemini inference
* Retrieval benchmarking
* Evaluation pipeline

Each request produces a complete execution trace that can be inspected inside Phoenix.

---

# Example Trace

```text
POST /chat
│
└── rag.query
    │
    ├── retrieval.semantic
    │      │
    │      ├── embedding.generate
    │      ├── retrieval.vector_search
    │      │      └── qdrant.query_points
    │      └── retrieval.reranking
    │
    ├── retrieval.context_builder
    ├── rag.build_prompt
    │
    └── GenerateContent
```

---

# Retrieval Analytics

Each retrieval operation records useful analytics that help evaluate retrieval quality.

Captured metrics include:

| Metric               | Description                              |
| -------------------- | ---------------------------------------- |
| Vector Search Time   | Time spent querying Qdrant               |
| Rerank Time          | Time spent by the Cross Encoder          |
| Total Retrieval Time | End-to-end retrieval latency             |
| Average Vector Score | Mean similarity score returned by Qdrant |
| Average Rerank Score | Mean Cross Encoder relevance score       |
| Candidate Count      | Number of retrieved candidates           |
| Returned Count       | Number of chunks after reranking         |
| Reranked Chunk Order | Final ranking after Cross Encoder        |

These metrics can be viewed directly inside Phoenix traces.

---

# Example Retrieval Attributes

```text
retrieval.strategy = semantic

retrieval.vector_search_time_ms = 21.47

retrieval.rerank_time_ms = 156.82

retrieval.total_retrieval_time_ms = 181.94

retrieval.average_vector_score = 0.811

retrieval.average_rerank_score = 0.947

retrieval.candidate_count = 20

retrieval.returned_count = 5

retrieval.reranker_model = BAAI/bge-reranker-base
```

---

# Benchmark Framework

A dedicated benchmarking framework is included to compare retrieval strategies objectively.

Instead of relying on subjective answer quality, retrieval performance is measured using information retrieval metrics.

The benchmark framework evaluates:

* Character Chunking
* Recursive Chunking
* Token Chunking
* Semantic Chunking

with and without Cross Encoder reranking.

---

# Supported Evaluation Metrics

## Recall@K

Measures whether the expected document exists within the first **K** retrieved results.

Higher is better.

---

## Precision@K

Measures how many of the retrieved documents are relevant.

Higher is better.

---

## Mean Reciprocal Rank (MRR)

Evaluates how early the correct document appears.

Earlier retrieval produces a higher score.

---

## NDCG

Normalized Discounted Cumulative Gain measures ranking quality while giving higher weight to documents appearing earlier in the ranking.

---

## Retrieval Latency

Measures retrieval speed including:

* embedding generation
* vector search
* reranking

This allows quality improvements to be evaluated alongside latency trade-offs.

---

# Benchmark Workflow

```text
Benchmark Dataset
        │
        ▼
Question
        │
        ▼
Retrieve
        │
        ▼
Evaluate
        │
        ▼
Calculate Metrics
        │
        ▼
Aggregate Results
```

---

# Benchmark Output

Each benchmark produces statistics for every retrieval strategy.

Example:

| Strategy          | Recall@5 | Precision@5 | MRR | NDCG | Avg Latency |
| ----------------- | -------: | ----------: | --: | ---: | ----------: |
| Character         |        - |           - |   - |    - |           - |
| Recursive         |        - |           - |   - |    - |           - |
| Token             |        - |           - |   - |    - |           - |
| Semantic          |        - |           - |   - |    - |           - |
| Semantic + Rerank |        - |           - |   - |    - |           - |

Replace these placeholder values with your benchmark results after running the evaluation.

---

# Why Benchmarking Matters

Changing a retrieval algorithm does not necessarily improve retrieval quality.

Every modification should be measurable.

The benchmark framework makes it possible to answer questions such as:

* Does semantic chunking outperform recursive chunking?
* Does reranking improve Precision@5?
* Does reranking increase MRR?
* What latency cost does reranking introduce?
* Which retrieval strategy performs best for a given dataset?

This encourages data-driven improvements instead of subjective assumptions.

---

# Technology Stack

## Backend

* FastAPI
* Python

---

## Vector Database

* Qdrant

---

## Embeddings

* Sentence Transformers

---

## Reranking

* Sentence Transformers CrossEncoder
* BAAI/bge-reranker-base

---

## Large Language Model

* Google Gemini

---

## Observability

* OpenTelemetry
* Arize Phoenix

---

## Evaluation

* Custom Benchmark Framework
* Recall@K
* Precision@K
* MRR
* NDCG

---

# Project Structure

```text
app/
│
├── api/
│
├── core/
│
├── infrastructure/
│     └── vector_store/
│
├── observability/
│
├── schemas/
│
├── services/
│     │
│     ├── evaluation/
│     │     └── benchmark/
│     ├── generation/
│     ├── ingestion/
│     ├── retrieval/
│     └── reranking/
│
└── main.py
```

The project follows a layered architecture separating:

* API layer
* Business logic
* Infrastructure
* Evaluation
* Observability

This keeps the retrieval pipeline modular and extensible.

---

# Configuration

Configuration is managed through environment variables.

Important options include:

```env
QDRANT_HOST=localhost

QDRANT_PORT=6333

GEMINI_API_KEY=your_api_key

PHOENIX_ENABLED=true

PHOENIX_PROJECT_NAME=observable-rag-backend

PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317

PHOENIX_PROTOCOL=grpc

PHOENIX_BATCH_EXPORT=true

PHOENIX_CAPTURE_CONTENT=true

INGESTION_FOLDER=data/ingestion

RERANKER_ENABLED=false

RERANKER_MODEL=bge

RERANKER_BGE_MODEL_NAME=BAAI/bge-reranker-base

RERANKER_CANDIDATE_COUNT=20

RERANKER_FINAL_TOP_K=5
```

The retrieval pipeline can therefore be modified without changing application code.

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<your-username>/observable-rag-backend.git

cd observable-rag-backend
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file.

Configure:

* Google Gemini API Key
* Qdrant host and port
* Phoenix collector endpoint

Start the application.

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

Phoenix UI:

```text
http://localhost:6006
```

---

---

# Running the Project

## 1. Start Qdrant

If using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## 2. Start Phoenix

```bash
phoenix serve
```

Phoenix UI:

```text
http://localhost:6006
```

---

## 3. Start FastAPI

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# Typical Workflow

The complete workflow of the project is shown below.

```text
Start Services
      │
      ▼
Ingest PDFs
      │
      ▼
Generate Embeddings
      │
      ▼
Store in Qdrant
      │
      ▼
Ask Questions
      │
      ▼
Retrieve
      │
      ▼
(Optional) Rerank
      │
      ▼
Generate Response
      │
      ▼
Inspect Phoenix Traces
      │
      ▼
Run Benchmarks
      │
      ▼
Compare Retrieval Strategies
```

---

# API Endpoints

## Ingestion

| Endpoint                       | Description                      |
| ------------------------------ | -------------------------------- |
| POST `/ingest`                 | Upload and ingest a PDF          |
| POST `/ingest/folder`          | Ingest every PDF inside a folder |
| DELETE `/collections/delete`   | Delete all Qdrant collections    |

---

## Chat

| Endpoint     | Description                                        |
| ------------ | -------------------------------------------------- |
| POST `/chat` | Ask questions using Retrieval-Augmented Generation |

Supports:

* Chunk strategy selection
* Optional reranking
* Configurable retrieval

---

## Benchmark

| Endpoint                 | Description               |
| ------------------------ | ------------------------- |
| POST `/benchmark/default` | Execute benchmark dataset |

---

## Retrieval Evaluation

| Endpoint                     | Description                |
| ---------------------------- | -------------------------- |
| POST `/evaluate/retrieval`   | Evaluate retrieval quality |

---

## Chunking

| Endpoint                   | Description                        |
| -------------------------- | ---------------------------------- |
| GET `/chunking/strategies` | List supported chunking strategies |

---

# Example Chat Request

```http
POST /chat
```

```json
{
  "query": "What is RSI?",
  "chunk_strat": "semantic"
}
```

---

# Example Response

```json
{
  "response": "Relative Strength Index (RSI) is a momentum oscillator..."
}
```

---

# Example Benchmark Request

```http
POST /benchmark/default?reranking_enabled=true&reranker_model=bge&candidate_count=20
```

---

# Example Retrieval Evaluation Request

```http
POST /evaluate/retrieval
```

```json
{
  "query": "Explain support and resistance.",
  "top_k": 5
}
```

---

# Phoenix Observability

Phoenix provides complete visibility into the retrieval pipeline.

Example trace:

```text
POST /chat
│
└── rag.query
    │
    ├── retrieval.semantic
    │      │
    │      ├── embedding.generate
    │      ├── retrieval.vector_search
    │      └── retrieval.reranking
    │
    ├── retrieval.context_builder
    ├── rag.build_prompt
    │
    └── GenerateContent
```

---

## Suggested Screenshots

Replace these placeholders with screenshots from your project.

```markdown
docs/images/

phoenix-trace.png

phoenix-reranking.png

benchmark-results.png

swagger-chat.png

swagger-benchmark.png
```

Then include them in the README.

Example:

```markdown
## Phoenix Trace

![Phoenix Trace](docs/images/phoenix-trace.png)
```

---

# Engineering Decisions

## Why Multiple Chunking Strategies?

Different documents benefit from different chunking techniques.

Instead of assuming one strategy performs best, the project benchmarks all supported strategies.

---

## Why Strategy Pattern?

Chunking and reranking are implemented using the Strategy Pattern to keep the retrieval pipeline extensible.

Adding a new chunker or reranker requires implementing a single interface without modifying retrieval logic.

---

## Why Cross Encoder Reranking?

Embedding similarity is efficient but not always optimal for ranking.

Cross Encoders jointly process the query and document, producing significantly better relevance estimates.

The trade-off is additional latency.

---

## Why Separate Benchmarking?

Retrieval quality should be measured objectively rather than relying on subjective answer quality.

Benchmarking provides measurable evidence for retrieval improvements.

---

## Why Phoenix?

Production AI systems require observability.

Phoenix enables inspection of:

* Retrieval latency
* Retrieved documents
* Prompt construction
* Model execution
* Reranking behavior

without adding manual logging throughout the codebase.

---

# Future Roadmap

This project is designed to evolve toward a production-grade AI retrieval platform.

Planned improvements include:

### Retrieval

* Hybrid Search (BM25 + Dense Retrieval)
* Parent-Child Retrieval
* Threshold Retrieval
* Contextual Retrieval
* Contextual Compression
* Metadata Filtering
* Self Query Retrieval

---

### Query Enhancement

* Multi Query Retrieval
* Query Rewriting
* HyDE
* Query Expansion

---

### Evaluation

* LLM-as-a-Judge
* Answer Faithfulness
* Context Precision
* Context Recall
* Hallucination Detection

---

### Observability

* Retrieval dashboards
* Latency dashboards
* Prompt comparison
* Cost tracking
* Token usage analytics

---

### Agentic AI

* Tool Calling
* Function Calling
* Multi-Agent Retrieval
* Planning Agents
* Memory Integration

---

# What I Learned

Building this project provided practical experience with modern AI Engineering concepts including:

* Production Retrieval-Augmented Generation
* Vector databases
* Embedding models
* Cross Encoder reranking
* Information Retrieval metrics
* Retrieval benchmarking
* OpenTelemetry
* Arize Phoenix
* FastAPI application architecture
* Modular software design
* Strategy Pattern
* AI system observability

The project emphasizes measurable retrieval quality, modular architecture, and production-oriented engineering practices.

---

# Repository Highlights

✔ Production-style FastAPI architecture

✔ Modular ingestion pipeline

✔ Four chunking strategies

✔ Cross Encoder reranking

✔ Retrieval benchmarking

✔ Recall@K, Precision@K, MRR, NDCG

✔ Phoenix observability

✔ OpenTelemetry instrumentation

✔ Configurable retrieval pipeline

✔ Extensible Strategy Pattern architecture

---

# License

This project is released under the MIT License.

---

# Acknowledgements

This project builds upon the excellent open-source ecosystem surrounding:

* FastAPI
* Qdrant
* Sentence Transformers
* Hugging Face
* Google Gemini
* OpenTelemetry
* Arize Phoenix

Special thanks to the maintainers of these libraries for making modern AI engineering more accessible.

---

# Connect

If you found this project interesting or have suggestions for improving the retrieval pipeline, feel free to open an issue or start a discussion.

Contributions, ideas, and feedback are always welcome.
