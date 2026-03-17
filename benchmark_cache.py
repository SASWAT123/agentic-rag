"""
Redis cache benchmark: measures latency for cache miss vs cache hit.
Run: uv run benchmark_cache.py
"""

import time
from dotenv import load_dotenv
from cache import RedisSemanticCache
from rag import build_agent

load_dotenv()

QUERIES = [
    "Who is Dobby?",
    "What is the Sorting Hat?",
    "Who is Voldemort?",
]


def run_benchmark():
    print("=" * 58)
    print(" Redis Semantic Cache — Performance Benchmark")
    print("=" * 58)

    # Flush old cache entries for a clean benchmark
    import redis as redis_lib
    r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
    r.delete("hp_rag_cache")
    print("\nCache cleared for clean run.\n")

    cache = RedisSemanticCache()
    agent = build_agent()

    results = []

    for query in QUERIES:
        # ── MISS: first call, nothing cached ──
        t0 = time.perf_counter()
        cached = cache.lookup(query)
        if not cached:
            result = agent.invoke({"messages": [("human", query)]})
            answer = result["messages"][-1].content
            cache.update(query, answer)
        miss_latency = time.perf_counter() - t0

        # ── HIT: same query again ──
        t0 = time.perf_counter()
        cached = cache.lookup(query)
        hit_latency = time.perf_counter() - t0

        results.append((query, miss_latency, hit_latency))
        print(f"  Query   : {query!r}")
        print(f"  MISS    : {miss_latency:.3f}s  (embed + agent + LLM + tools + cache.update)")
        print(f"  HIT     : {hit_latency:.3f}s  (embed + Redis fetch + cosine similarity)")
        print(f"  Speedup : {miss_latency / hit_latency:.1f}x faster\n")

    print("=" * 58)
    avg_miss = sum(r[1] for r in results) / len(results)
    avg_hit  = sum(r[2] for r in results) / len(results)
    print(f" Avg MISS latency : {avg_miss:.3f}s")
    print(f" Avg HIT latency  : {avg_hit:.3f}s")
    print(f" Avg speedup      : {avg_miss / avg_hit:.1f}x")
    print("=" * 58)

    cache.close()


if __name__ == "__main__":
    run_benchmark()
