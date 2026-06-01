# ResearchCrew — Multi-Agent Research System

> Build spec for Cursor/Claude agents. Follow this document to build and deploy the project.

## Purpose

Give it a topic, and 4 specialized AI agents (Planner, Searcher, Analyst, Synthesizer) autonomously research the web, analyze findings, and produce a structured report with citations. Built with LangGraph state machines, tool-calling, and streaming output showing each agent's reasoning in real-time.

**Problem it solves:** Research is tedious — searching, reading, cross-referencing, synthesizing. ResearchCrew automates the entire workflow while showing its reasoning, so you can verify and steer the process.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              LANGGRAPH STATE MACHINE                     │
│                                                          │
│  ┌──────────┐                                            │
│  │ Planner  │  Decomposes query into subtasks            │
│  └────┬─────┘                                            │
│       │                                                  │
│  ┌────┴────────┐                                         │
│  │   Router    │  Routes subtasks by type                │
│  └──┬──────┬───┘                                         │
│     │      │                                             │
│     ▼      ▼                                             │
│  ┌──────┐ ┌────────┐                                     │
│  │Search│ │Analyst │  Parallel execution                 │
│  │Agent │ │Agent   │                                     │
│  └──┬───┘ └───┬────┘                                     │
│     │         │                                          │
│     ▼         ▼                                          │
│  ┌──────────────┐                                        │
│  │ Loop Check   │  Max 3 iterations                      │
│  └──────┬───────┘                                        │
│         │                                                │
│    ┌────┴────┐                                           │
│    │continue │──► back to Planner                        │
│    │done     │──► Synthesizer                            │
│    └─────────┘                                           │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐                                         │
│  │ Synthesizer │  Produces structured report              │
│  └─────────────┘                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Structured Report + Sources + Reasoning Trace

TOOLS:
  ├── web_search     (Tavily)
  ├── fetch_url      (Trafilatura)
  └── note_taker     (in-memory persistent notes)
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Agent framework | LangGraph 0.3.x |
| LLM orchestration | LangChain 0.3.x |
| LLM | OpenAI `gpt-4o-mini` OR Google `gemini-2.5-flash` (user-configured) |
| Web search | Tavily API |
| URL extraction | Trafilatura |
| API server | FastAPI + WebSocket |
| Frontend | Streamlit |
| Python version | 3.11+ |

## Directory Structure

```
researchcrew/
├── src/
│   ├── __init__.py
│   ├── config.py                    # Environment + provider config
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py               # Query decomposition
│   │   ├── searcher.py              # Web search agent
│   │   ├── analyst.py               # Document analysis agent
│   │   └── synthesizer.py           # Report synthesis agent
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py                 # ResearchState definition
│   │   ├── nodes.py                 # Graph node functions
│   │   ├── edges.py                 # Conditional edge logic
│   │   └── builder.py              # Graph construction + compile
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── web_search.py            # Tavily search wrapper
│   │   ├── url_fetcher.py           # URL content extraction
│   │   └── note_taker.py            # Persistent note storage
│   └── api/
│       ├── __init__.py
│       ├── main.py                  # FastAPI app
│       ├── routes.py                # REST + WebSocket endpoints
│       └── schemas.py               # Request/response models
├── ui/
│   └── app.py                       # Streamlit interface
├── tests/
│   ├── test_agents.py
│   ├── test_tools.py
│   ├── test_graph.py
│   └── test_integration.py
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

## Implementation Guide

### 1. LangGraph State (`src/graph/state.py`)

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class Subtask(TypedDict):
    type: str          # "search" or "analyze"
    query: str
    priority: int      # 1 = highest
    status: str        # "pending", "done"

class SearchResult(TypedDict):
    query: str
    title: str
    url: str
    content: str
    relevance: float

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str                          # Original user query
    plan: list[Subtask]                 # Decomposed subtasks
    search_results: list[SearchResult]  # Raw search findings
    analyzed_findings: list[dict]       # Processed analyses
    notes: list[str]                    # Accumulated research notes
    report: str                         # Final synthesized report
    sources: list[dict]                 # Bibliography
    current_step: str                   # For UI progress display
    iteration: int                      # Loop counter (max 3)
```

### 2. Agent Implementations (`src/agents/`)

**planner.py:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class PlannerAgent:
    """Decomposes a research query into actionable subtasks."""

    def __init__(self, llm):
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", PLANNER_PROMPT),
                ("human", "{input}"),
            ])
            | llm
            | JsonOutputParser()
        )

    def plan(self, query: str, existing_findings: list[str] | None = None) -> list[dict]:
        context = ""
        if existing_findings:
            context = f"\n\nFindings so far:\n" + "\n".join(existing_findings)
        result = self.chain.invoke({"input": f"Research query: {query}{context}"})
        return result["subtasks"]

PLANNER_PROMPT = """You are a research planner. Break the user's research query into
specific, actionable subtasks.

Output JSON:
{
    "subtasks": [
        {"type": "search", "query": "specific search query", "priority": 1},
        {"type": "analyze", "query": "what to analyze from search results", "priority": 2}
    ],
    "reasoning": "Why these subtasks cover the research question"
}

Rules:
- 3-6 subtasks maximum
- "search" tasks find information; "analyze" tasks synthesize/compare findings
- Order by priority (1 = most important)
- Be specific in search queries — vague queries get poor results"""
```

**searcher.py:**
```python
from langchain.agents import create_react_agent, AgentExecutor

class SearcherAgent:
    """Searches the web and extracts key information."""

    def __init__(self, llm, tools: list):
        self.agent = AgentExecutor(
            agent=create_react_agent(llm, tools, SEARCHER_PROMPT),
            tools=tools,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def search(self, query: str) -> list[dict]:
        result = self.agent.invoke({"input": query})
        return result

SEARCHER_PROMPT = """You are a research agent. Use the available tools to find
information relevant to the query. For each source you find:
1. Search for the topic
2. Fetch the most promising URLs for deeper content
3. Extract key facts, data points, and quotes
4. Note the source URL for citation

Be thorough but focused. Stop when you have 3-5 solid sources."""
```

**analyst.py:**
```python
class AnalystAgent:
    """Analyzes search results and extracts structured insights."""

    def __init__(self, llm):
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", ANALYST_PROMPT),
                ("human", "{input}"),
            ])
            | llm
            | JsonOutputParser()
        )

    def analyze(self, search_results: list[dict], query: str) -> dict:
        formatted = "\n\n".join(
            f"Source: {r['url']}\nContent: {r['content'][:1000]}"
            for r in search_results
        )
        return self.chain.invoke({
            "input": f"Query: {query}\n\nSearch Results:\n{formatted}"
        })

ANALYST_PROMPT = """You are a research analyst. Given search results, extract structured insights.

Output JSON:
{
    "key_findings": ["finding 1", "finding 2"],
    "data_points": [{"fact": "...", "source": "url"}],
    "contradictions": ["any conflicting information"],
    "gaps": ["what's still unclear or missing"],
    "needs_more_research": true/false,
    "follow_up_queries": ["if needs_more_research, what to search next"]
}"""
```

**synthesizer.py:**
```python
class SynthesizerAgent:
    """Synthesizes all findings into a structured research report."""

    def __init__(self, llm):
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYNTHESIZER_PROMPT),
                ("human", "{input}"),
            ])
            | llm
            | StrOutputParser()
        )

    def synthesize(self, query: str, findings: list[dict],
                   notes: list[str], sources: list[dict]) -> str:
        findings_text = "\n\n".join(str(f) for f in findings)
        notes_text = "\n".join(notes) if notes else "No additional notes."
        sources_text = "\n".join(f"- {s['url']}: {s.get('title', '')}" for s in sources)

        return self.chain.invoke({
            "input": f"Query: {query}\n\nFindings:\n{findings_text}\n\n"
                     f"Notes:\n{notes_text}\n\nSources:\n{sources_text}"
        })

SYNTHESIZER_PROMPT = """You are a research report writer. Synthesize all findings into
a well-structured report.

Format:
## Executive Summary
[2-3 sentences answering the research question]

## Key Findings
[Numbered list of main findings with source citations]

## Detailed Analysis
[In-depth discussion organized by theme/subtopic]

## Gaps & Limitations
[What we couldn't fully answer, conflicting information]

## Sources
[Numbered bibliography with URLs]

Rules:
- Cite sources inline as [1], [2], etc.
- Be objective — present multiple viewpoints when they exist
- Distinguish between facts and opinions
- Flag low-confidence claims"""
```

### 3. LangGraph Construction (`src/graph/`)

**nodes.py:**
```python
from ..agents.planner import PlannerAgent
from ..agents.searcher import SearcherAgent
from ..agents.analyst import AnalystAgent
from ..agents.synthesizer import SynthesizerAgent

def planner_node(state: ResearchState) -> dict:
    planner = PlannerAgent(state["llm"])
    existing = [str(f) for f in state.get("analyzed_findings", [])]
    plan = planner.plan(state["query"], existing or None)
    return {"plan": plan, "current_step": "planning"}

def searcher_node(state: ResearchState) -> dict:
    searcher = SearcherAgent(state["llm"], state["tools"])
    search_tasks = [t for t in state["plan"] if t["type"] == "search" and t["status"] == "pending"]
    results = []
    for task in search_tasks:
        result = searcher.search(task["query"])
        results.append(result)
        task["status"] = "done"
    return {"search_results": results, "current_step": "searching", "plan": state["plan"]}

def analyst_node(state: ResearchState) -> dict:
    analyst = AnalystAgent(state["llm"])
    analysis = analyst.analyze(state["search_results"], state["query"])
    return {
        "analyzed_findings": state.get("analyzed_findings", []) + [analysis],
        "current_step": "analyzing",
    }

def synthesizer_node(state: ResearchState) -> dict:
    synthesizer = SynthesizerAgent(state["llm"])
    report = synthesizer.synthesize(
        state["query"], state["analyzed_findings"],
        state.get("notes", []), state.get("sources", []),
    )
    return {"report": report, "current_step": "synthesizing"}
```

**edges.py:**
```python
def should_search_or_synthesize(state: ResearchState) -> str:
    pending_searches = [t for t in state["plan"] if t["type"] == "search" and t["status"] == "pending"]
    if pending_searches:
        return "searcher"
    return "analyst"

def needs_more_research(state: ResearchState) -> str:
    if state.get("iteration", 0) >= 3:
        return "done"
    latest = state.get("analyzed_findings", [])
    if latest and latest[-1].get("needs_more_research", False):
        return "continue"
    return "done"
```

**builder.py:**
```python
from langgraph.graph import StateGraph, START, END
from .state import ResearchState
from .nodes import planner_node, searcher_node, analyst_node, synthesizer_node
from .edges import should_search_or_synthesize, needs_more_research

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        should_search_or_synthesize,
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
```

### 4. Tools (`src/tools/`)

**web_search.py:**
```python
from langchain_community.tools.tavily_search import TavilySearchResults

def create_search_tool(api_key: str):
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_raw_content=True,
        api_key=api_key,
    )
```

**url_fetcher.py:**
```python
from langchain_core.tools import tool
import trafilatura

@tool
def fetch_url(url: str) -> str:
    """Fetch and extract main content from a URL. Use this to read full articles."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        content = trafilatura.extract(downloaded, include_links=True)
        return content[:5000] if content else "Could not extract content"
    return "Failed to fetch URL"
```

**note_taker.py:**
```python
from langchain_core.tools import tool

_notes: list[str] = []

@tool
def add_note(note: str) -> str:
    """Save an important finding or observation for later synthesis."""
    _notes.append(note)
    return f"Note saved. Total notes: {len(_notes)}"

@tool
def get_notes() -> str:
    """Retrieve all saved research notes."""
    if not _notes:
        return "No notes saved yet."
    return "\n".join(f"{i+1}. {n}" for i, n in enumerate(_notes))

def clear_notes():
    _notes.clear()
```

### 5. API with WebSocket Streaming (`src/api/`)

**routes.py:**
```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json

router = APIRouter()

@router.post("/research")
async def start_research(request: ResearchRequest):
    """Start a research task (blocking, returns final report)."""
    pass

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

        graph = build_research_graph()
        async for event in graph.astream(
            {"query": query, "iteration": 0},
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                await websocket.send_json({
                    "event": "step",
                    "agent": node_name,
                    "step": node_output.get("current_step", ""),
                    "content": str(node_output),
                })

        # Send final report
        await websocket.send_json({"event": "complete"})
    except WebSocketDisconnect:
        pass
```

### 6. Streamlit UI (`ui/app.py`)

Layout:
- **Sidebar:** Settings (LLM provider, API key, Tavily key)
- **Main area:**
  - Research query input
  - "Research" button
  - Real-time agent activity display (expandable per agent)
  - Final report with markdown rendering
  - Sources with clickable links
  - Download report as markdown

```python
import streamlit as st
import websockets
import asyncio
import json

st.set_page_config(page_title="ResearchCrew", page_icon="🔬", layout="wide")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM Provider", ["gemini", "openai"])
    api_key = st.text_input("LLM API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")

st.title("ResearchCrew")
st.caption("Multi-agent research system. Enter a topic and watch agents work.")

query = st.text_area("Research Topic", placeholder="e.g., Compare RAG vs fine-tuning for domain-specific LLM applications")

if st.button("Research", type="primary", disabled=not query or not api_key):
    progress_container = st.container()
    report_container = st.container()

    with progress_container:
        st.subheader("Agent Activity")
        # WebSocket connection to stream agent steps
        # Display each agent's work in expanders:
        # - Planner: show subtasks
        # - Searcher: show search queries and results
        # - Analyst: show key findings
        # - Synthesizer: show report being written

    with report_container:
        st.subheader("Research Report")
        # Render final report as markdown
        # Show sources
        # Download button
```

## Environment Variables

```env
# LLM (configurable via UI settings page)
LLM_PROVIDER=gemini
LLM_API_KEY=your-api-key
LLM_MODEL=gemini-2.5-flash

# Search
TAVILY_API_KEY=your-tavily-key

# Server
API_HOST=0.0.0.0
API_PORT=8000
```

## Dependencies (`requirements.txt`)

```
langgraph>=0.3.0
langchain>=0.3.0
langchain-openai>=0.3.0
langchain-google-genai>=2.0.0
langchain-community>=0.3.0
tavily-python>=0.5.0
trafilatura>=1.12.0
fastapi>=0.115.0
uvicorn>=0.30.0
websockets>=13.0
streamlit>=1.40.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/darsigangothri06/researchcrew.git
cd researchcrew

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Environment (optional — can use UI settings instead)
cp .env.example .env

# 5. Start API server
uvicorn src.api.main:app --reload --port 8000

# 6. Start UI (separate terminal)
streamlit run ui/app.py --server.port 8501

# Open:
# API docs: http://localhost:8000/docs
# Research UI: http://localhost:8501
```

## Testing

```bash
# Run tests
pytest tests/ -v

# Test full pipeline
python -m src.graph.builder --query "What are the latest advances in AI agents?"
```

## Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## GitHub Repository Setup

> **CRITICAL — READ BEFORE ANY GIT OPERATION**
>
> This project MUST be pushed to the **personal** GitHub account ONLY.
> - **CORRECT account:** `darsigangothri06` (gangothri.darsi@gmail.com)
> - **DO NOT USE:** `gangothri-bryt` / `gangothri@bryt.in` — this is the company work account. NEVER push personal projects to the work account.
> - **DO NOT modify global git config** (`--global`). Only set LOCAL config inside this repo.
> - **VERIFY before every push:** Run `git config user.name && git config user.email` and confirm it shows `darsigangothri06` / `gangothri.darsi@gmail.com`. If it doesn't, STOP and fix it.

```bash
# 1. Create repo on GitHub first
gh repo create darsigangothri06/researchcrew --public --description "Multi-agent research system with LangGraph"

# 2. Initialize local repo
git init

# 3. SET LOCAL GIT IDENTITY (NOT --global)
git config user.name "darsigangothri06"
git config user.email "gangothri.darsi@gmail.com"

# 4. VERIFY identity before proceeding
git config user.name   # Must show: darsigangothri06
git config user.email  # Must show: gangothri.darsi@gmail.com

# 5. Add remote and push
git remote add origin https://github.com/darsigangothri06/researchcrew.git
git add .
git commit -m "feat: ResearchCrew — multi-agent research system with LangGraph"
git push -u origin main
```

If `gh` CLI is authenticated as the work account, authenticate personal account first:
```bash
gh auth login  # Choose: github.com → HTTPS → Login with browser → authenticate as darsigangothri06
```
