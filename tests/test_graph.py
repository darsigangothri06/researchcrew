from src.graph.edges import should_search_or_analyze, needs_more_research
from src.graph.builder import build_research_graph


def test_should_search_when_pending_tasks():
    state = {
        "plan": [
            {"type": "search", "query": "test", "priority": 1, "status": "pending"},
        ]
    }
    assert should_search_or_analyze(state) == "searcher"


def test_should_analyze_when_no_pending_searches():
    state = {
        "plan": [
            {"type": "search", "query": "test", "priority": 1, "status": "done"},
            {"type": "analyze", "query": "analyze test", "priority": 2, "status": "pending"},
        ]
    }
    assert should_search_or_analyze(state) == "analyst"


def test_should_analyze_when_empty_plan():
    state = {"plan": []}
    assert should_search_or_analyze(state) == "analyst"


def test_needs_more_research_done_after_max_iterations():
    state = {"iteration": 3, "analyzed_findings": [{"needs_more_research": True}]}
    assert needs_more_research(state) == "done"


def test_needs_more_research_continue():
    state = {
        "iteration": 1,
        "analyzed_findings": [
            {"needs_more_research": True, "follow_up_queries": ["more info"]}
        ],
    }
    assert needs_more_research(state) == "continue"


def test_needs_more_research_done():
    state = {
        "iteration": 1,
        "analyzed_findings": [{"needs_more_research": False}],
    }
    assert needs_more_research(state) == "done"


def test_graph_compiles():
    graph = build_research_graph()
    assert graph is not None
