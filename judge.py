"""
LLM-as-a-Judge: evaluates every agent response for faithfulness and relevance.

- Faithfulness (1-5): Is the answer grounded in the retrieved context?
                      Does it avoid making claims not supported by sources?
- Relevance   (1-5): Does the answer actually address what was asked?

Verdict: PASS if both scores >= 3, else FAIL.
The judge receives the original question, the agent's answer, and the raw
tool results (retrieved book chunks / web results) that the agent used.
"""

import json
import textwrap
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

load_dotenv()

JUDGE_MODEL = "gpt-4o"

JUDGE_PROMPT = """You are a strict evaluator assessing the quality of an AI answer about Harry Potter.

You will be given:
1. QUESTION  — what the user asked
2. ANSWER    — what the AI responded
3. CONTEXT   — the source material the AI retrieved (book passages or web results)

Score the answer on two dimensions:

FAITHFULNESS (1-5):
  5 — Every claim is directly supported by the context
  4 — Most claims supported; minor unsupported details
  3 — Core claims supported but some additions not in context
  2 — Several claims contradict or go beyond the context
  1 — Answer is largely hallucinated or contradicts the context

RELEVANCE (1-5):
  5 — Directly and completely answers the question
  4 — Answers the question with minor gaps
  3 — Partially answers; misses some key aspects
  2 — Loosely related but doesn't properly answer
  1 — Does not answer the question at all

Respond with ONLY valid JSON in this exact format:
{
  "faithfulness_score": <int 1-5>,
  "faithfulness_reason": "<one sentence>",
  "relevance_score": <int 1-5>,
  "relevance_reason": "<one sentence>",
  "verdict": "<PASS or FAIL>",
  "verdict_reason": "<one sentence overall summary>"
}"""


@dataclass
class JudgeResult:
    faithfulness_score: int
    faithfulness_reason: str
    relevance_score: int
    relevance_reason: str
    verdict: str
    verdict_reason: str

    def display(self) -> None:
        W = 58  # inner content width (chars between the two │ borders)

        def row(text: str = "") -> str:
            """Pad or truncate text to exactly W chars and wrap with borders."""
            if len(text) > W:
                text = text[: W - 1] + "…"
            return f"  │{text:<{W}}│"

        def score_bar(score: int) -> str:
            """ASCII score bar: [#####] 5/5 — always 12 chars wide."""
            return "[" + "#" * score + "-" * (5 - score) + f"] {score}/5"

        def wrapped_rows(text: str, indent: int = 3) -> list[str]:
            """Wrap long text into multiple padded rows."""
            prefix = " " * indent
            lines = textwrap.wrap(text, width=W - indent)
            return [row(prefix + line) for line in lines] if lines else [row()]

        verdict_label = "PASS" if self.verdict == "PASS" else "FAIL"
        sep = "─" * W

        print()
        print(f"  ┌{sep}┐")
        print(row(f" Judge Verdict : {verdict_label}"))
        print(f"  ├{sep}┤")
        print(row(f" Faithfulness  : {score_bar(self.faithfulness_score)}"))
        for r in wrapped_rows(self.faithfulness_reason):
            print(r)
        print(row())
        print(row(f" Relevance     : {score_bar(self.relevance_score)}"))
        for r in wrapped_rows(self.relevance_reason):
            print(r)
        print(f"  ├{sep}┤")
        for r in wrapped_rows(self.verdict_reason, indent=1):
            print(r)
        print(f"  └{sep}┘")


def _extract_context(messages: list[BaseMessage]) -> str:
    """Pull tool call results out of the agent message history."""
    parts = []
    for msg in messages:
        # LangGraph ToolMessages carry the tool output in .content
        if hasattr(msg, "type") and msg.type == "tool":
            parts.append(msg.content)
    return "\n\n---\n\n".join(parts) if parts else "No context retrieved."


def judge_response(
    question: str,
    answer: str,
    messages: list[BaseMessage],
) -> JudgeResult:
    """Run the judge LLM and return a structured JudgeResult."""
    context = _extract_context(messages)

    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CONTEXT:\n{context}"
    )

    response = llm.invoke([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user",   "content": user_content},
    ])

    raw = response.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(raw)

    # Enforce verdict based on scores (guard against LLM inconsistency)
    scores_pass = data["faithfulness_score"] >= 3 and data["relevance_score"] >= 3
    data["verdict"] = "PASS" if scores_pass else "FAIL"

    return JudgeResult(**data)
