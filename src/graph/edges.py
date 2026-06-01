from __future__ import annotations

from .state import ResearchState

MAX_ITERATIONS = 3


def should_search_or_analyze(state: ResearchState) -> str:
    pending_searches = [
        t for t in state.get("plan", [])
        if t.get("type") == "search" and t.get("status") == "pending"
    ]
    if pending_searches:
        return "searcher"
    return "analyst"


def needs_more_research(state: ResearchState) -> str:
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "done"

    latest = state.get("analyzed_findings", [])
    if latest and latest[-1].get("needs_more_research", False):
        follow_ups = latest[-1].get("follow_up_queries", [])
        if follow_ups:
            return "continue"

    return "done"
