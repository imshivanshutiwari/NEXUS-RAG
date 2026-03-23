# ⚡ NEXUS-RAG

**Production-Grade Agentic RAG System** — LangGraph · AWS Bedrock Claude-3 · pgvector HNSW · ColBERT · RAGAS

---

## Architecture

```
Data Sources → Ingestion → pgvector + BM25 → Hybrid Retrieval
      ↓                                            ↓
  6-Node LangGraph Pipeline:              CrossEncoder + ColBERT
  Router → Retriever → Reranker              Reranking
     → Generator → Evaluator ←→ Healer         ↓
                                        AWS Bedrock Claude-3
                                        RAGAS Inline Evaluation
                                        Self-Healing Loop
```

### Key Features
- **5 Real Data Sources**: Wikipedia, ArXiv, SEC EDGAR, PubMed, CommonCrawl
- **Hybrid Search**: BM25 (rank-bm25) + pgvector HNSW (cosine) + RRF fusion
- **3-Stage Reranking**: HybridRetriever → CrossEncoder → ColBERT MaxSim
- **HyDE + Multi-Query Expansion** via Bedrock
- **RAGAS Inline Evaluation** (faithfulness, relevancy, precision, recall)
- **Self-Healing Loop**: up to 3 reformulation attempts
- **Evidently AI Drift Detection** with auto re-index trigger
- **Dash Dashboard**: 22 interactive Plotly visualizations
- **FastAPI Server** with SSE streaming
- **WebSocket Hub** for real-time updates

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/imshivanshutiwari/NEXUS-RAG.git
cd NEXUS-RAG
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start infrastructure
docker-compose up -d

# 4. Ingest corpus
make fetch
make ingest

# 5. Run API
make api

# 6. Run Dashboard
make dashboard
```

---

## Project Structure

```
NEXUS-RAG/
├── configs/          # YAML configuration files
├── data/
│   ├── fetchers/     # Wikipedia, ArXiv, SEC EDGAR, PubMed, CommonCrawl
│   └── processors/   # PDF, chunking, metadata, document store
├── ingestion/        # Embedder, pgvector indexer, BM25, deduplicator
├── retrieval/        # Dense, sparse, hybrid, CrossEncoder, ColBERT, HyDE
├── agents/           # LangGraph: Router→Retriever→Reranker→Generator→Evaluator→Healer
├── generation/       # Bedrock client, prompt builder, citation builder
├── evaluation/       # RAGAS, faithfulness, answer relevancy, context metrics, benchmark
├── monitoring/       # Drift detector, latency tracker, quality monitor, alert manager
├── streaming/        # SSE query streamer, pipeline events, WebSocket hub
├── pipeline/         # End-to-end orchestrator
├── api/              # FastAPI server + routes
├── dashboard/        # Dash app with 22 Plotly visualizations
├── tests/            # 30 pytest tests
└── notebooks/        # EDA, retrieval analysis, evaluation analysis
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `COHERE_API_KEY` | Cohere Embed v3 API key |
| `AWS_ACCESS_KEY_ID` | AWS credentials for Bedrock |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `AWS_DEFAULT_REGION` | AWS region (default: us-east-1) |
| `POSTGRES_HOST` | PostgreSQL host |
| `POSTGRES_DB` | Database name (default: nexusrag) |
| `POSTGRES_USER` | Database user |
| `POSTGRES_PASSWORD` | Database password |

---

## Makefile Targets

```
make fetch      # Fetch all 5 data sources
make ingest     # Run ingestion pipeline
make api        # Start FastAPI server (port 8000)
make dashboard  # Start Dash dashboard (port 8050)
make bench      # Run RAGAS benchmark
make test       # Run pytest
make lint       # Run flake8 + black
make docker     # Start docker-compose services
```

---

## Security

All dependencies are pinned to patched versions:
- `aiohttp>=3.13.3` (zip bomb fix)
- `black>=26.3.1` (arbitrary file write fix)
- `langchain-community>=0.3.27` (XXE + pickle deserialization fixes)
- `mlflow>=3.9.0` (RCE, directory traversal, command injection fixes)

---

## License

MIT
