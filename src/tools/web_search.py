from langchain_core.tools import tool
from tavily import TavilyClient


def create_search_tool(api_key: str):
    client = TavilyClient(api_key=api_key)

    @tool
    def web_search(query: str) -> list[dict]:
        """Search the web for information on a topic. Returns a list of results with url and content."""
        response = client.search(query, search_depth="advanced", max_results=5)
        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]

    return web_search
