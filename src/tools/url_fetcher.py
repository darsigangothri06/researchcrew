from langchain_core.tools import tool
import trafilatura


@tool
def fetch_url(url: str) -> str:
    """Fetch and extract main content from a URL. Use this to read full articles."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            content = trafilatura.extract(downloaded, include_links=True)
            return content[:5000] if content else "Could not extract content from this URL."
        return "Failed to fetch URL — the server did not respond."
    except Exception as e:
        return f"Error fetching URL: {e}"
