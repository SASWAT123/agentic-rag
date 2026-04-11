"""
Harry Potter RAG Agent — interactive CLI.

Usage:
    uv run main.py                          # no cache
    uv run main.py --cache faiss            # FAISS semantic cache (no TTL)
    uv run main.py --cache redis            # Redis cache, default TTL 300s
    uv run main.py --cache redis --ttl 60   # Redis cache, TTL 60s
    uv run main.py --cache redis --ttl 0    # Redis cache, no expiry
"""

import argparse
from rag import setup_phoenix_tracing
from orchestrator import build_orchestrator
from cache import get_cache
from judge import judge_response


def parse_args():
    parser = argparse.ArgumentParser(description="Harry Potter RAG Agent")
    parser.add_argument(
        "--cache",
        choices=["faiss", "redis", "none"],
        default="none",
        help="Semantic cache backend (default: none)",
    )
    parser.add_argument(
        "--ttl",
        type=int,
        default=300,
        help="Redis cache TTL in seconds — 0 means no expiry (default: 300)",
    )
    return parser.parse_args()


def _ttl_label(cache_backend: str, ttl: int) -> str:
    if cache_backend != "redis":
        return cache_backend
    if ttl == 0:
        return "redis (no expiry)"
    minutes, seconds = divmod(ttl, 60)
    if minutes and seconds:
        return f"redis | TTL {minutes}m{seconds}s"
    elif minutes:
        return f"redis | TTL {minutes}m"
    return f"redis | TTL {seconds}s"


def main():
    args = parse_args()

    setup_phoenix_tracing()
    orchestrator = build_orchestrator()
    cache = get_cache(args.cache, ttl=args.ttl)

    label = _ttl_label(args.cache, args.ttl)
    print("\n╔══════════════════════════════════════╗")
    print("║   Harry Potter RAG Agent              ║")
    print(f"║   Cache: {label:<28}║")
    print("║   Type 'quit' or 'exit' to stop       ║")
    print("╚══════════════════════════════════════╝\n")

    try:
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break

            # Check cache first
            if cache:
                cached_answer = cache.lookup(query)
                if cached_answer:
                    print(f"\nAgent (from cache): {cached_answer}\n")
                    continue

            # Cache miss — invoke the orchestrator
            answer, messages, route = orchestrator.invoke(query)

            route_label = {"harry_potter": "HP Agent", "other_chars": "Others Agent", "both": "Both Agents"}.get(route, route)
            print(f"\n[Orchestrator] Routed to: {route_label}")
            print(f"\nAgent: {answer}\n")

            # Judge the response (skip for cache hits — answer was already judged)
            print("[Judge] Evaluating response...", end="", flush=True)
            verdict = judge_response(query, answer, messages)
            print("\r", end="")  # clear the "Evaluating..." line
            verdict.display()

            # Only cache responses that pass the judge
            if cache:
                if verdict.verdict == "PASS":
                    cache.update(query, answer)
                else:
                    print("  [Cache] Response not cached — judge returned FAIL.\n")
    finally:
        if cache:
            cache.close()


if __name__ == "__main__":
    main()
