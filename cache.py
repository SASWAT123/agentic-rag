"""
Semantic caching backends: FAISS and Redis.

Both embed the user query with text-embedding-3-small and use cosine similarity
to find sufficiently close past queries (threshold: SIMILARITY_THRESHOLD).
Caching is at the full query → answer level — a cache hit skips the agent entirely.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SIMILARITY_THRESHOLD = 0.92
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def _embed(text: str, client: OpenAI) -> np.ndarray:
    """Embed text and return a unit-normalized float32 vector."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    vec /= np.linalg.norm(vec)  # normalize → cosine similarity via inner product
    return vec


class SemanticCache(ABC):
    """Common interface for all semantic cache backends."""

    @abstractmethod
    def lookup(self, query: str) -> Optional[str]:
        """Return cached answer if a sufficiently similar query exists, else None."""

    @abstractmethod
    def update(self, query: str, answer: str) -> None:
        """Store a new query→answer pair in the cache."""

    @abstractmethod
    def close(self) -> None:
        """Flush/close any resources."""


# ─────────────────────────────── FAISS backend ────────────────────────────────

class FAISSSemanticCache(SemanticCache):
    """
    Persists query embeddings in a local FAISS IndexFlatIP index.
    IndexFlatIP does exact inner-product search — equivalent to cosine similarity
    on unit-normalized vectors. Fast, zero infrastructure, survives restarts.
    """

    _CACHE_DIR = Path(__file__).parent / "faiss_cache"
    _INDEX_FILE = _CACHE_DIR / "cache.index"
    _DATA_FILE = _CACHE_DIR / "cache.json"

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        self._oai = OpenAI()
        self._threshold = threshold
        self._CACHE_DIR.mkdir(exist_ok=True)

        if self._INDEX_FILE.exists() and self._DATA_FILE.exists():
            self._index = faiss.read_index(str(self._INDEX_FILE))
            with open(self._DATA_FILE) as f:
                self._entries: list[dict] = json.load(f)
        else:
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._entries = []

        print(f"[FAISS Cache] Ready — {self._index.ntotal} entries loaded from disk.")

    def lookup(self, query: str) -> Optional[str]:
        if self._index.ntotal == 0:
            print("[FAISS Cache] MISS — cache is empty.")
            return None

        vec = _embed(query, self._oai).reshape(1, -1)
        scores, indices = self._index.search(vec, 1)
        score = float(scores[0][0])
        idx = int(indices[0][0])

        if score >= self._threshold:
            print(f"[FAISS Cache] HIT  (similarity={score:.4f}, "
                  f"matched: {self._entries[idx]['query']!r})")
            return self._entries[idx]["answer"]

        print(f"[FAISS Cache] MISS (best similarity={score:.4f})")
        return None

    def update(self, query: str, answer: str) -> None:
        vec = _embed(query, self._oai).reshape(1, -1)
        self._index.add(vec)
        self._entries.append({"query": query, "answer": answer})
        self._persist()

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self._INDEX_FILE))
        with open(self._DATA_FILE, "w") as f:
            json.dump(self._entries, f, indent=2)

    def close(self) -> None:
        self._persist()


# ─────────────────────────────── Redis backend ────────────────────────────────

class RedisSemanticCache(SemanticCache):
    """
    Stores each cache entry as an individual Redis key: hp_rag_cache:{uuid}.
    Each key carries its own TTL — Redis expires and deletes it automatically.
    On lookup, SCAN finds all live (non-expired) keys, then cosine similarity
    selects the best match above the threshold.

    TTL=0 means no expiry (entries live forever).
    Works with plain Redis — no Redis Stack needed.
    """

    _PREFIX = "hp_rag_cache"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        threshold: float = SIMILARITY_THRESHOLD,
        ttl: int = 300,  # seconds; 0 = no expiry
    ):
        import redis
        import uuid as _uuid

        self._oai = OpenAI()
        self._threshold = threshold
        self._ttl = ttl
        self._uuid = _uuid
        self._redis = redis.Redis(host=host, port=port, decode_responses=True)

        try:
            self._redis.ping()
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Redis at {host}:{port}. "
                "Make sure Redis is running (e.g. `redis-server`)."
            ) from e

        count = sum(1 for _ in self._redis.scan_iter(f"{self._PREFIX}:*"))
        ttl_label = f"{ttl}s" if ttl > 0 else "none"
        print(f"[Redis Cache] Ready — {count} live entries | TTL: {ttl_label}")

    def _live_keys(self) -> list[str]:
        """Return all non-expired cache keys."""
        return list(self._redis.scan_iter(f"{self._PREFIX}:*"))

    def lookup(self, query: str) -> Optional[str]:
        keys = self._live_keys()
        if not keys:
            print("[Redis Cache] MISS — cache is empty.")
            return None

        query_vec = _embed(query, self._oai)
        best_score = -1.0
        best_entry: Optional[dict] = None

        for key in keys:
            raw = self._redis.get(key)
            if raw is None:
                continue  # expired between scan and get
            entry = json.loads(raw)
            cached_vec = np.array(entry["embedding"], dtype=np.float32)
            score = float(np.dot(query_vec, cached_vec))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self._threshold and best_entry:
            ttl_remaining = self._redis.ttl(f"{self._PREFIX}:{best_entry['id']}")
            ttl_info = f", TTL remaining: {ttl_remaining}s" if ttl_remaining > 0 else ""
            print(f"[Redis Cache] HIT  (similarity={best_score:.4f}, "
                  f"matched: {best_entry['query']!r}{ttl_info})")
            return best_entry["answer"]

        print(f"[Redis Cache] MISS (best similarity={best_score:.4f})")
        return None

    def update(self, query: str, answer: str) -> None:
        entry_id = str(self._uuid.uuid4())
        key = f"{self._PREFIX}:{entry_id}"
        entry = {"id": entry_id, "query": query, "embedding": _embed(query, self._oai).tolist(), "answer": answer}
        serialized = json.dumps(entry)
        if self._ttl > 0:
            self._redis.setex(key, self._ttl, serialized)
        else:
            self._redis.set(key, serialized)

    def close(self) -> None:
        pass  # Redis persists automatically


# ──────────────────────────────── Factory ─────────────────────────────────────

def get_cache(backend: str, ttl: int = 300) -> Optional[SemanticCache]:
    """Return the selected cache backend, or None if caching is disabled."""
    if backend == "faiss":
        return FAISSSemanticCache()
    if backend == "redis":
        return RedisSemanticCache(ttl=ttl)
    print("[Cache] Disabled.")
    return None
