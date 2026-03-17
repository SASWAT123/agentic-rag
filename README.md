# Harry Potter RAG Agent

An **Agentic Retrieval-Augmented Generation (RAG)** system that answers questions about the Harry Potter universe using all 15 books as a knowledge base. When the books don't contain the answer, the agent automatically falls back to a live web search.

Built with Python, LangChain, LangGraph, OpenAI, FAISS, Arize Phoenix, and Redis.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Semantic Caching](#semantic-caching)
- [LLM-as-a-Judge](#llm-as-a-judge)
- [Observability with Phoenix](#observability-with-phoenix)
- [Rebuilding the Vector Index](#rebuilding-the-vector-index)
- [Cache Benchmark](#cache-benchmark)
- [Component Reference](#component-reference)

---

## Architecture

```
User Query
    │
    ▼
Semantic Cache (FAISS or Redis)
    │ HIT → return cached answer instantly (~200ms)
    │ MISS ↓
    ▼
LangGraph ReAct Agent (GPT-4o)
    │
    ├─► Tool: search_books
    │       Embeds query → FAISS similarity search
    │       Returns top-5 chunks from 15 HP books
    │       with book title + page citations
    │
    └─► Tool: search_web  (fallback if books have no answer)
            DuckDuckGo search → returns results + URLs
    │
    ▼
LLM-as-a-Judge (GPT-4o)
    Evaluates: Faithfulness + Relevance (each 1–5)
    Verdict: PASS → store in cache | FAIL → skip cache
    │
    ▼
Answer displayed to user
```

---

## Features

| Feature | Details |
|---|---|
| Knowledge base | 15 Harry Potter PDFs (HP 1–8, companion books, short stories) |
| Embedding model | `text-embedding-3-small` (1536 dimensions) |
| Vector store | FAISS `IndexFlatL2` — 10,072 chunks, exact search |
| LLM | GPT-4o (`temperature=0`) |
| Web fallback | DuckDuckGo — triggered automatically when books don't have the answer |
| Semantic cache | FAISS (local) or Redis (distributed) with configurable TTL |
| Cache strategy | Full query→answer level — cache hits skip all LLM + tool calls |
| LLM judge | GPT-4o scores every response on Faithfulness + Relevance |
| Observability | Arize Phoenix — full traces, token usage, latency at `localhost:6006` |
| Cache speedup | ~18x faster on hits (200ms vs 4s) |

---

## Project Structure

```
harry-potter-rag/
├── main.py              # CLI entry point — arg parsing, main loop
├── rag.py               # LangGraph agent, Phoenix tracing, tools
├── ingest.py            # PDF loading, chunking, FAISS index builder
├── cache.py             # FAISSSemanticCache + RedisSemanticCache
├── judge.py             # LLM-as-a-Judge (faithfulness + relevance)
├── benchmark_cache.py   # Latency benchmark: cache miss vs hit
├── faiss_index/         # Persisted book vector index (auto-generated)
│   ├── index.faiss      # 59MB — raw embedding vectors
│   └── index.pkl        # 10MB — document text + metadata
├── faiss_cache/         # Persisted query cache (auto-generated)
│   ├── cache.index      # FAISS index for cached queries
│   └── cache.json       # Cached query→answer pairs
├── pyproject.toml       # Dependencies (managed by uv)
├── uv.lock              # Locked dependency versions
└── .env                 # API keys (not committed)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Managed by `uv` |
| [uv](https://docs.astral.sh/uv/) | latest | Package manager |
| OpenAI API key | — | Needs access to `gpt-4o` and `text-embedding-3-small` |
| Redis | 7+ | Only required for `--cache redis` |

**Install Redis (macOS):**
```bash
brew install redis
brew services start redis
```

**Install Redis (Ubuntu/Debian):**
```bash
sudo apt install redis-server
sudo systemctl start redis
```

---

## Installation

**1. Clone the repository and enter the project:**
```bash
cd harry-potter-rag
```

**2. Install dependencies:**
```bash
uv sync
```

**3. Copy the environment file and add your API key:**
```bash
cp .env.example .env
```
Edit `.env`:
```
OPENAI_API_KEY=sk-...your-key-here...
```

**4. Build the vector index** (one-time, ~1–2 minutes, costs ~$0.01 in OpenAI embeddings):
```bash
uv run ingest.py
```
This loads all 15 PDFs, splits them into 10,072 chunks, embeds each with `text-embedding-3-small`, and saves the FAISS index to `faiss_index/`. You never need to run this again unless you add new books.

---

## Usage

### Basic — no caching
```bash
uv run main.py
```

### With FAISS semantic cache
```bash
uv run main.py --cache faiss
```

### With Redis semantic cache (default TTL: 5 minutes)
```bash
uv run main.py --cache redis
```

### With Redis and custom TTL
```bash
uv run main.py --cache redis --ttl 60     # 1 minute
uv run main.py --cache redis --ttl 3600   # 1 hour
uv run main.py --cache redis --ttl 0      # no expiry
```

### Example session
```
You: Who is Dobby?

Agent: Dobby is a house-elf who was originally in the service of the Malfoy
family. Harry Potter set Dobby free...
(Source: Goblet of Fire, p.390; Order of the Phoenix, p.399)

  ┌──────────────────────────────────────────────────────────┐
  │ Judge Verdict : PASS                                     │
  ├──────────────────────────────────────────────────────────┤
  │ Faithfulness  : [#####] 5/5                              │
  │   All claims are directly supported by the context.      │
  │                                                          │
  │ Relevance     : [#####] 5/5                              │
  │   The answer directly and completely addresses the query. │
  ├──────────────────────────────────────────────────────────┤
  │ The answer is faithful and fully relevant.               │
  └──────────────────────────────────────────────────────────┘

You: Who played Dumbledore in the movies?

[Redis Cache] MISS (best similarity=0.31)

Agent: I couldn't find this in the books, so I searched the web.
Richard Harris played Dumbledore in the first two films. After his death,
Michael Gambon took over from Prisoner of Azkaban onwards.
(Source: https://en.wikipedia.org/wiki/Albus_Dumbledore)

You: quit
Goodbye!
```

---

## Semantic Caching

The cache stores past `query → answer` pairs. On each new query, the query is embedded and compared against stored embeddings using **cosine similarity**. If the score exceeds the threshold, the cached answer is returned immediately — no LLM or tool calls are made.

### How it works

```
New query → embed (OpenAI API) → compare against stored embeddings
    similarity ≥ 0.92 → HIT  → return cached answer (~200ms)
    similarity < 0.92 → MISS → run agent (~4s) → store result
```

### FAISS vs Redis

| | FAISS Cache | Redis Cache |
|---|---|---|
| Storage | `faiss_cache/` on disk | Redis server (`localhost:6379`) |
| Lookup | FAISS in-process (very fast) | Fetch all + numpy dot products |
| TTL support | No (manual workaround needed) | Yes — native per-key TTL via `SETEX` |
| Infrastructure | None | Requires Redis running |
| Best for | Local dev, single process | Production, multi-process, distributed |
| Speedup | ~18x vs no cache | ~18x vs no cache |

### Similarity threshold

Controlled by `SIMILARITY_THRESHOLD = 0.92` in `cache.py`.

| Threshold | Effect |
|---|---|
| `0.97+` | Only near-identical queries hit cache |
| `0.92` | Same intent, different phrasing — hits cache (default) |
| `0.85` | Broader matching — more hits, slight risk of wrong answer |

### Cache + Judge integration

Only responses that **pass the judge** are stored in the cache. This ensures that hallucinated or irrelevant answers are never served from cache to future users.

### Viewing the Redis cache

Install [RedisInsight](https://redis.io/insight/) (already installed if you followed setup):
```bash
open "/Applications/Redis Insight.app"
```
Connect to `localhost:6379`. Each cache entry is stored as a key `hp_rag_cache:{uuid}` with a visible TTL countdown.

Useful Redis CLI commands:
```bash
redis-cli KEYS "hp_rag_cache:*"    # list all cache keys
redis-cli LLEN hp_rag_cache        # count entries
redis-cli DEL $(redis-cli KEYS "hp_rag_cache:*")  # clear cache
```

---

## LLM-as-a-Judge

Every agent response (excluding cache hits) is automatically evaluated by a second GPT-4o call acting as a judge.

### Scoring dimensions

| Dimension | What it checks |
|---|---|
| **Faithfulness** (1–5) | Is the answer grounded in retrieved context? Does it avoid hallucinating facts not in the source? |
| **Relevance** (1–5) | Does the answer actually address what was asked? |

### Verdict

- **PASS** — both Faithfulness ≥ 3 and Relevance ≥ 3 → answer is shown and cached
- **FAIL** — either score < 3 → answer is shown but **not cached**

### Score guide

| Score | Meaning |
|---|---|
| 5 | Every claim supported / directly answers the question |
| 4 | Minor gaps or unsupported details |
| 3 | Core is correct / partially answers |
| 2 | Several issues |
| 1 | Hallucinated / does not answer |

---

## Observability with Phoenix

[Arize Phoenix](https://arize.com/docs/phoenix) runs locally and captures full traces of every agent interaction.

Phoenix starts automatically when you run `main.py`. Open the UI at:
```
http://localhost:6006
```

### What you can see per trace

- The full prompt sent to GPT-4o
- Which tool was called (`search_books` or `search_web`) and with what arguments
- The raw tool output (retrieved book chunks or web results)
- The final LLM response
- Token usage and latency for every step

---

## Rebuilding the Vector Index

If you add new PDFs to the `Harry Potter/` folder, rebuild the index:
```bash
uv run ingest.py
```
This re-embeds everything from scratch and overwrites `faiss_index/`. The old index is replaced.

To inspect the current index without rebuilding:
```bash
uv run python -c "
from ingest import load_vectorstore
vs = load_vectorstore()
print(f'Vectors: {vs.index.ntotal}')
print(f'Dimensions: {vs.index.d}')
"
```

---

## Cache Benchmark

Run the included benchmark to measure cache miss vs hit latency across 3 sample queries:

```bash
uv run benchmark_cache.py
```

Sample output:
```
  Query   : 'Who is Dobby?'
  MISS    : 4.511s
  HIT     : 0.273s
  Speedup : 16.6x faster

  Avg MISS latency : 4.172s
  Avg HIT latency  : 0.234s
  Avg speedup      : 17.9x
```

---

## Component Reference

| File | Responsibility |
|---|---|
| `ingest.py` | `load_all_books()` · `chunk_documents()` · `build_vectorstore()` · `load_vectorstore()` |
| `rag.py` | `setup_phoenix_tracing()` · `build_agent()` · `search_books` tool · `search_web` tool |
| `cache.py` | `FAISSSemanticCache` · `RedisSemanticCache` · `get_cache(backend, ttl)` |
| `judge.py` | `judge_response(question, answer, messages)` · `JudgeResult.display()` |
| `main.py` | CLI arg parsing · main query loop · cache + judge orchestration |
| `benchmark_cache.py` | Standalone Redis cache latency benchmark |

---

## CLI Reference

```
uv run main.py [--cache {faiss,redis,none}] [--ttl SECONDS]

Options:
  --cache    Cache backend to use (default: none)
             faiss  — local FAISS semantic cache, no TTL
             redis  — Redis semantic cache with TTL
             none   — no caching
  --ttl      Redis cache TTL in seconds (default: 300)
             0 means entries never expire
```
