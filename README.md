<div align="center">

# NimbusMart Support Agent

### A retrieval-augmented customer support agent built with Semantic Kernel, ChromaDB and FastAPI

**[HERE AND NOW AI](https://hereandnowai.com)** — *AI is Good*

</div>

---

## Overview

NimbusMart Support Agent is a role-aware customer support assistant for a fictional e-commerce
store. It answers questions from a curated knowledge base of policies, FAQs and product
documentation, and grounds every answer in retrieved source material rather than free-form
generation.

The retrieval stack combines **hybrid search** — dense embeddings via ChromaDB alongside BM25
lexical matching — with an **LLM reranker** that reorders candidates by relevance before they
reach the answering model.

> **Project status: work in progress.** The retrieval, chunking, auth and session layers are
> implemented. The FastAPI HTTP surface (`app/main.py`) is not wired up yet, so the React
> frontend cannot talk to the backend as-is. See [Roadmap](#roadmap).

## Architecture

```
frontend/          React 18 + TypeScript + Vite chat UI
backend/
  app/
    config.py        Environment + path configuration
    auth.py          Token-based login, customer vs. staff roles
    session.py       Per-conversation state, citations, tool calls
    kernel_setup.py  Azure OpenAI chat + embedding wiring
    rag/
      chunking.py       Markdown-header-aware splitting (700 chars, 100 overlap)
      reranker.py       LLM-based candidate reranking
      retriever_types.py
    data/
      documents/     Knowledge base (policies, FAQs, catalog)
      orders.json    Demo order records
      users.json     Demo accounts
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | Semantic Kernel |
| LLM & embeddings | Azure OpenAI |
| Vector store | ChromaDB |
| Lexical search | BM25 (`rank-bm25`) |
| Chunking | LangChain text splitters |
| API | FastAPI + Uvicorn |
| Frontend | React 18, TypeScript, Vite |
| Package manager | `uv` (Python), `npm` (Node) |

## Getting Started

### Prerequisites

- Python 3.13+ and [`uv`](https://docs.astral.sh/uv/)
- Node.js 18+
- An Azure OpenAI resource with a chat deployment and an embedding deployment

### Backend

```bash
cd backend
cp .env.example .env      # then fill in your Azure OpenAI credentials
uv sync
```

### Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

## Configuration

All secrets live in `backend/.env`, which is gitignored. Copy `.env.example` and supply your own
values:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | API key for your Azure OpenAI resource |
| `AZURE_OPENAI_ENDPOINT` | Resource endpoint URL |
| `AZURE_OPENAI_MODEL` | Chat completion deployment name |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Embedding deployment name |
| `AZURE_OPENAI_API_VERSION` | API version string |
| `AZURE_OPENAI_JUDGE_DEPLOYMENT` | Deployment used for reranking / evaluation |

## Demo Accounts

`backend/app/data/users.json` ships with **hardcoded demo credentials** so the role-based access
logic can be exercised locally. They are sample fixtures for a fictional store — not real
accounts — and this file should be replaced with a proper user store before any real deployment.

| Username | Role |
|---|---|
| `riya`, `arjun`, `kavya` | customer |
| `admin`, `manager` | staff |

## Roadmap

- [ ] Wire up `app/main.py` with FastAPI chat, login and health routes
- [ ] Connect the React frontend to the live backend
- [ ] Persist the ChromaDB index and add an ingestion command
- [ ] Rename `app/orders_data.py.py` (has a duplicated extension)
- [ ] Add tests for chunking and reranking

## License

Released under the [MIT License](LICENSE).

---

<div align="center">

**HERE AND NOW AI**

[Website](https://hereandnowai.com) · *AI is Good*

</div>
