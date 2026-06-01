from __future__ import annotations

import json
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from .schemas import ResearchRequest, ResearchResponse, HealthResponse
from ..config import Settings
from ..tools.web_search import create_search_tool
from ..tools.note_taker import NoteStore
from ..graph.builder import build_research_graph

router = APIRouter()


def _build_config(settings_dict: dict) -> dict:
    settings = Settings(
        provider=settings_dict.get("provider", "gemini"),
        llm_api_key=settings_dict.get("llm_api_key", ""),
        tavily_api_key=settings_dict.get("tavily_api_key", ""),
        model=settings_dict.get("model", ""),
    )
    llm = settings.get_llm()
    search_tool = create_search_tool(settings.tavily_api_key)
    return {
        "configurable": {
            "llm": llm,
            "search_tool": search_tool,
        }
    }


@router.get("/")
async def root():
    return {
        "name": "ResearchCrew API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "research": "POST /research",
            "stream": "WS /research/stream",
        },
    }


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@router.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest):
    """Start a research task (blocking, returns final report)."""
    if not request.settings.llm_api_key:
        raise HTTPException(status_code=400, detail="llm_api_key is required")
    if not request.settings.tavily_api_key:
        raise HTTPException(status_code=400, detail="tavily_api_key is required")

    try:
        config = _build_config(request.settings.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid settings: {e}")

    graph = build_research_graph()

    initial_state = {
        "query": request.query,
        "iteration": 0,
        "messages": [],
        "plan": [],
        "search_results": [],
        "analyzed_findings": [],
        "notes": [],
        "report": "",
        "sources": [],
        "current_step": "",
    }

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {e}")

    return ResearchResponse(
        report=final_state.get("report", ""),
        sources=final_state.get("sources", []),
        iterations=final_state.get("iteration", 0),
    )


@router.websocket("/research/stream")
async def stream_research(websocket: WebSocket):
    """Stream research progress via WebSocket.

    Client sends: {"query": "...", "settings": {...}}
    Server streams: {"event": "step", "agent": "planner", "content": "..."}
    Final: {"event": "complete", "report": "...", "sources": [...]}
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data["query"]
        settings = data.get("settings", {})

        config = _build_config(settings)
        graph = build_research_graph()

        initial_state = {
            "query": query,
            "iteration": 0,
            "messages": [],
            "plan": [],
            "search_results": [],
            "analyzed_findings": [],
            "notes": [],
            "report": "",
            "sources": [],
            "current_step": "",
        }

        async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                step_data = {
                    "event": "step",
                    "agent": node_name,
                    "step": node_output.get("current_step", ""),
                    "content": _safe_serialize(node_output),
                }
                await websocket.send_json(step_data)

        final_state = await graph.ainvoke(initial_state, config=config)
        await websocket.send_json({
            "event": "complete",
            "report": final_state.get("report", ""),
            "sources": final_state.get("sources", []),
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "event": "error",
                "content": str(e),
            })
        except Exception:
            pass


def _safe_serialize(obj: dict) -> str:
    """Serialize dict to string, handling non-serializable values."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)
