from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from .state import ResearchState
from .nodes import planner_node, searcher_node, analyst_node, synthesizer_node
from .edges import should_search_or_analyze, needs_more_research


def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        should_search_or_analyze,
        {"searcher": "searcher", "analyst": "analyst"},
    )
    graph.add_edge("searcher", "analyst")
    graph.add_conditional_edges(
        "analyst",
        needs_more_research,
        {"continue": "planner", "done": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()
