# Enterprise Knowledge Engine

A local, production-grade Retrieval-Augmented Generation (RAG) pipeline built using LangGraph, Qdrant, and native Hugging Face Transformers. The architecture leverages hybrid search (dense semantic vectors combined with sparse lexical tokens) via a unified **BGE-M3** model, cross-encoder reranking through **BGE-Reranker-v2-M3**, and local in-memory text synthesis with **Qwen2.5-1.5B-Instruct**.

---

## Architecture Overview

The pipeline processes user queries through a structured, state-managed execution graph:

1. **Document Ingestion:** PDFs are processed using native extractors, split into clean structural segments, and embedded concurrently using dense and sparse encoders.
2. **Hybrid Vector Search:** Input queries extract top candidates from Qdrant using Reciprocal Rank Fusion (RRF) to merge keyword matching and dense embeddings.
3. **Cross-Encoder Reranking:** Candidates pass through an intensive attention verification node. Matches below a threshold score of `-3.0` are pruned, with a top-2 fallback guarantee to maintain generation stability.
4. **Synthesis Engine:** Validated context fragments feed an in-memory LLM generator configured for local inference.
5. **Continuous Evaluation:** A custom, isolated LLM-as-a-Judge execution layer maps pipeline outputs against synthetic ground-truth metrics (Context Precision, Faithfulness) to capture regression trends without external network overhead.

---

## Project Structure

```text
enterprise-knowledge-engine/
├── .github/workflows/          # CI/CD deployment configurations
├── backend/
│   └── app/
│       ├── api/
│       │   └── v1/             # API routing endpoints
│       │   ├── deps.py         # API dependency injection blocks
│       │   └── main.py         # FastAPI application entrypoint
│       ├── core/
│       │   ├── config.py       # Environment & service configurations
│       │   └── logging.py      # Structured logs engine configuration
│       ├── graph/
│       │   ├── nodes.py        # LangGraph workflow orchestration nodes
│       │   ├── state.py        # Centralized AgentState context schema
│       │   ├── tools.py        # Hybrid query utilities
│       │   └── workflow.py     # Graph topology definitions
│       ├── schemas/
│       │   ├── ingest.py       # Ingestion data validation models
│       │   └── query.py        # Query layer validation models
│       └── services/
│           ├── document_processor.py # PDF parsing and chunking engine
│           ├── llm_generation.py     # Local synthesis execution module
│           └── vector_store.py       # Qdrant hybrid storage client wrapper
├── evaluation/
│   ├── data/                   # Folder destination for source benchmark PDFs
│   ├── generate_test_set.py    # In-memory evaluation QA dataset generator
│   ├── index_eval_data.py      # Automatic database seeding module
│   ├── run_evals.py            # Automated validation suite harness
│   ├── test_set.json           # Generated evaluation QA dataset cache
│   └── eval_report.json        # Aggregated pipeline performance metrics output
├── frontend/
│   └── app.py                  # User interface presentation layer
├── tests/                      # Pytest automation suite components
├── .env                        # Environment variable secrets configurations
├── .gitignore                  # Git tracking exclusion configuration
├── docker-compose.yml          # Container configuration for local service daemons
├── Dockerfile.prod             # Production multi-stage image blueprint
└── requirements.txt            # Project application dependency requirements

```

---

## Quickstart & Verification Loop

### 1. Environment Setup

Initialize a Python virtual environment and ensure local requirements are fully configured:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Generate the Synthetic Evaluation Dataset

Place raw target verification files (PDFs) inside `evaluation/data/` and execute the pipeline dataset generator. This leverages your internal Hugging Face `LLMGenerationService` wrapper to form structured context-query blocks:

```powershell
python -m evaluation.generate_test_set

```

### 3. Run the Automated Evaluation Suite

Run the centralized test harness. This suite programmatically flushes your old database tables, reads the dynamic file paths written to `test_set.json`, seeds them through your production document parser, fires queries through the LangGraph orchestrator, and saves analytical metrics:

```powershell
python -m evaluation.run_evals

```

### 4. Inspect Performance Logs

Check the generated `evaluation/eval_report.json` document to view system performance metrics:

```json
{
    "metrics_summary": {
        "avg_faithfulness": 0.9000,
        "avg_context_precision": 1.0000,
        "total_evaluated_cases": 1
    },
    "detailed_runs": [
        {
            "query": "What was the new target range for the federal funds rate...",
            "ground_truth": "3-1/2 to 3-3/4 percent",
            "system_generation": "The new target range for the federal funds rate... was 3-1/2 to 3-3/4 percent.",
            "visited_steps": [
                "retrieve_node",
                "rerank_node",
                "generate_node"
            ],
            "scores": {
                "faithfulness": 0.9,
                "context_precision": 1.0
            },
            "critique": "The response accurately reflects the information provided..."
        }
    ]
}

```

---

## Configuration Details

* **Embedding Model:** BAAI/bge-m3 (Dense Dimension: 1024, Sparse Token Index on Disk)
* **Reranker Model:** BAAI/bge-reranker-v2-m3 (Max Length: 1024 tokens)
* **Synthesis Core:** Qwen/Qwen2.5-1.5B-Instruct
* **Database Logic:** Reciprocal Rank Fusion via primary dense vector matches and `"sparse-text"` indices.
