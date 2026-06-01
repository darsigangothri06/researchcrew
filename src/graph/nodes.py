from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ..agents.planner import PlannerAgent
from ..agents.searcher import SearcherAgent
from ..agents.analyst import AnalystAgent
from ..agents.synthesizer import SynthesizerAgent
from .state import ResearchState


def planner_node(state: ResearchState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    planner = PlannerAgent(llm)
    existing = [str(f) for f in state.get("analyzed_findings", [])]
    plan = planner.plan(state["query"], existing or None)
    return {"plan": plan, "current_step": "planning"}


def searcher_node(state: ResearchState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    search_tool = config["configurable"]["search_tool"]
    searcher = SearcherAgent(llm, search_tool)

    search_tasks = [
        t for t in state.get("plan", [])
        if t.get("type") == "search" and t.get("status") == "pending"
    ]

    all_results = list(state.get("search_results", []))
    updated_plan = list(state.get("plan", []))

    for task in search_tasks:
        findings = searcher.search(task["query"])
        for f in findings:
            all_results.append({
                "query": task["query"],
                "title": f.get("title", ""),
                "url": f.get("url", ""),
                "content": f.get("content", ""),
                "relevance": f.get("relevance", 0.5),
            })
        for t in updated_plan:
            if t["query"] == task["query"] and t["type"] == "search":
                t["status"] = "done"

    sources = [
        {"url": r["url"], "title": r["title"]}
        for r in all_results if r.get("url")
    ]

    return {
        "search_results": all_results,
        "sources": sources,
        "plan": updated_plan,
        "current_step": "searching",
    }


def analyst_node(state: ResearchState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    analyst = AnalystAgent(llm)
    analysis = analyst.analyze(
        state.get("search_results", []),
        state["query"],
    )
    existing = list(state.get("analyzed_findings", []))
    existing.append(analysis)
    iteration = state.get("iteration", 0) + 1
    return {
        "analyzed_findings": existing,
        "current_step": "analyzing",
        "iteration": iteration,
    }


def synthesizer_node(state: ResearchState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    synthesizer = SynthesizerAgent(llm)
    report = synthesizer.synthesize(
        state["query"],
        state.get("analyzed_findings", []),
        state.get("notes", []),
        state.get("sources", []),
    )
    return {"report": report, "current_step": "synthesizing"}
