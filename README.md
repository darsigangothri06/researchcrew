# ResearchCrew — Multi-Agent Research System

Give it a topic, and 4 specialized AI agents (Planner, Searcher, Analyst, Synthesizer) autonomously research the web, analyze findings, and produce a structured report with citations.

Built with **LangGraph** state machines, **LangChain** tool-calling, and **FastAPI** for the backend.

## Architecture

```
User Query → Planner → Searcher → Analyst → (loop if needed) → Synthesizer → Report
```

- **Planner**: Decomposes the query into subtasks
- **Searcher**: Searches the web via Tavily and extracts findings
- **Analyst**: Analyzes results, identifies gaps, decides if more research is needed
- **Synthesizer**: Produces a structured report with citations

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent Framework | LangGraph 0.3.x |
| LLM | OpenAI (gpt-4o-mini) or Google (gemini-2.5-flash) |
| Web Search | Tavily API |
| URL Extraction | Trafilatura |
| API Server | FastAPI + WebSocket |
| Frontend | Streamlit |

## Quick Start

```bash
# Clone
git clone https://github.com/darsigangothri06/researchcrew.git
cd researchcrew

# Virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements.txt

# Start API server
uvicorn src.api.main:app --reload --port 8000

# Start UI (separate terminal)
streamlit run ui/app.py --server.port 8501
```

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Research (blocking)
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest advances in AI agents?",
    "settings": {
      "provider": "gemini",
      "llm_api_key": "YOUR_KEY",
      "tavily_api_key": "YOUR_TAVILY_KEY"
    }
  }'
```

## Environment Variables

Copy `.env.example` to `.env` or configure via the Streamlit UI sidebar:

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `gemini` or `openai` |
| `LLM_API_KEY` | API key for the LLM provider |
| `TAVILY_API_KEY` | Tavily search API key |

## License

MIT
