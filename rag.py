"""
RAG agent: Phoenix tracing + FAISS book search + DuckDuckGo web fallback.
"""

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import phoenix as px

from ingest import load_vectorstore

load_dotenv()

LLM_MODEL = "gpt-4o"
RETRIEVER_K = 5
RELEVANCE_THRESHOLD = 0.75  # minimum cosine similarity to consider a result useful

SYSTEM_PROMPT = """You are an expert on the Harry Potter universe.

Follow these rules strictly:
1. ALWAYS use the `search_books` tool first to look up the answer in the Harry Potter books.
2. If `search_books` returns no useful results, THEN use the `search_web` tool.
3. When answering from the books, cite the exact book title and page number from the results.
4. When answering from the web, clearly state: "I couldn't find this in the books, so I searched the web." Then cite the URL source.
5. Never make up information. Only answer based on tool results."""


def setup_phoenix_tracing():
    """Start a local Phoenix server and instrument LangChain with OpenTelemetry."""
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")

    exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    LangChainInstrumentor().instrument(tracer_provider=provider)


def _make_tools():
    vectorstore = load_vectorstore()

    @tool
    def search_books(query: str) -> str:
        """Search the Harry Potter books for information. Always use this tool first.
        Returns relevant passages with book title and page number citations."""
        results = vectorstore.similarity_search_with_relevance_scores(query, k=RETRIEVER_K)
        relevant = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]
        if not relevant:
            return "No relevant passages found in the books."
        parts = []
        for doc, _ in relevant:
            book = doc.metadata.get("book", "Unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[Source: {book}, Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    web_search = DuckDuckGoSearchResults(
        name="search_web",
        description=(
            "Search the web for Harry Potter information NOT found in the books. "
            "Only use this if search_books returned no useful results. "
            "Returns results with titles, snippets, and URLs."
        ),
        num_results=4,
    )

    return [search_books, web_search]


def build_agent(system_prompt: str = SYSTEM_PROMPT):
    """Build a LangGraph ReAct agent with book search and web fallback."""
    tools = _make_tools()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return create_react_agent(llm, tools, prompt=system_prompt)


if __name__ == "__main__":
    setup_phoenix_tracing()
    agent = build_agent()

    query = "What is the Patronus charm and how does Harry learn it?"
    print(f"\nQuery: {query}\n")
    result = agent.invoke({"messages": [("human", query)]})
    print(f"\nAnswer:\n{result['messages'][-1].content}")
