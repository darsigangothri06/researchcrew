from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class Subtask(TypedDict):
    type: str       # "search" or "analyze"
    query: str
    priority: int   # 1 = highest
    status: str     # "pending", "done"


class SearchResult(TypedDict):
    query: str
    title: str
    url: str
    content: str
    relevance: float


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    plan: list[Subtask]
    search_results: list[SearchResult]
    analyzed_findings: list[dict]
    notes: list[str]
    report: str
    sources: list[dict]
    current_step: str
    iteration: int
