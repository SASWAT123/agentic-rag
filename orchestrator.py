"""
Orchestrator: classifies incoming queries and routes them to the appropriate
specialist agent(s). Synthesizes responses when both agents are needed.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from hp_agent import build_hp_agent
from others_agent import build_others_agent
from rag import LLM_MODEL

# ── Routing ──────────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a query classifier for a Harry Potter RAG system.

Given a user query, classify it into exactly one of these categories:
- "harry_potter"  — the query is specifically and solely about Harry Potter the character
- "other_chars"   — the query is about characters other than Harry Potter (Hermione, Ron, Dumbledore, Voldemort, etc.)
- "both"          — the query involves Harry Potter AND other characters, OR it is ambiguous / general (plot, world-building, spells, etc.)

Return ONLY the category string. No explanation, no punctuation — just one of: harry_potter, other_chars, both"""

VALID_ROUTES = {"harry_potter", "other_chars", "both"}

# ── Synthesis ─────────────────────────────────────────────────────────────────

SYNTHESIZER_SYSTEM_PROMPT = """You are a response synthesizer for a Harry Potter question-answering system.

You will receive:
- The original user question
- A response from the Harry Potter specialist (focused on Harry Potter the character)
- A response from the Other Characters specialist (focused on all other characters)

Your job is to merge these two responses into a single, coherent, well-structured answer.
Eliminate any redundancy, preserve all citations (book title + page number), and present
the information in a logical order. Do not add any information that wasn't in the two responses."""


class Orchestrator:
    def __init__(self):
        print("  [Orchestrator] Loading HP agent...")
        self.hp_agent = build_hp_agent()
        print("  [Orchestrator] Loading Others agent...")
        self.others_agent = build_others_agent()
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    def _classify(self, query: str) -> str:
        """Return 'harry_potter', 'other_chars', or 'both'."""
        response = self.llm.invoke([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", query),
        ])
        route = response.content.strip().lower()
        if route not in VALID_ROUTES:
            # Fallback: if LLM returns something unexpected, call both
            return "both"
        return route

    def _run_agent(self, agent, query: str) -> tuple[str, list[BaseMessage]]:
        result = agent.invoke({"messages": [("human", query)]})
        messages = result["messages"]
        return messages[-1].content, messages

    def invoke(self, query: str) -> tuple[str, list[BaseMessage], str]:
        """
        Route the query and return (answer, messages, route).

        - answer   : final answer string to display / judge / cache
        - messages : full message list for the judge (combined when both agents run)
        - route    : 'harry_potter' | 'other_chars' | 'both'
        """
        route = self._classify(query)

        if route == "harry_potter":
            answer, messages = self._run_agent(self.hp_agent, query)
            return answer, messages, route

        if route == "other_chars":
            answer, messages = self._run_agent(self.others_agent, query)
            return answer, messages, route

        # "both" — run both agents and synthesize
        print("  [Orchestrator] Routing to both agents...")
        hp_answer, hp_messages = self._run_agent(self.hp_agent, query)
        others_answer, others_messages = self._run_agent(self.others_agent, query)

        synthesis = self.llm.invoke([
            ("system", SYNTHESIZER_SYSTEM_PROMPT),
            ("human", (
                f"Question: {query}\n\n"
                f"Harry Potter Specialist:\n{hp_answer}\n\n"
                f"Other Characters Specialist:\n{others_answer}"
            )),
        ])

        # Combine message histories so the judge has full tool-call context
        combined_messages = hp_messages + others_messages
        return synthesis.content, combined_messages, route


def build_orchestrator() -> Orchestrator:
    """Construct and return the orchestrator (loads both specialist agents)."""
    return Orchestrator()
