from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

SEARCHER_PROMPT = """You are a research search agent. Use the search results provided \
to extract the most relevant information for the research query.

Given search results, extract and organize the key information.

Output JSON:
{{
    "findings": [
        {{
            "title": "descriptive title",
            "url": "source url",
            "content": "key information extracted",
            "relevance": 0.9
        }}
    ]
}}

Rules:
- Extract 3-5 most relevant findings
- Include the source URL for each finding
- Rate relevance from 0.0 to 1.0
- Focus on facts, data points, and expert opinions
- Return ONLY valid JSON, no markdown fences"""


class SearcherAgent:
    """Searches the web and extracts key information."""

    def __init__(self, llm, search_tool):
        self.llm = llm
        self.search_tool = search_tool
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", SEARCHER_PROMPT),
                ("human", "{input}"),
            ])
            | llm
            | JsonOutputParser()
        )

    def search(self, query: str) -> list[dict]:
        raw_results = self.search_tool.invoke(query)

        if isinstance(raw_results, str):
            raw_results = [{"content": raw_results, "url": "N/A"}]

        formatted = "\n\n".join(
            f"URL: {r.get('url', 'N/A')}\nContent: {r.get('content', '')[:1500]}"
            for r in raw_results
        )

        result = self.chain.invoke({
            "input": f"Query: {query}\n\nSearch Results:\n{formatted}"
        })
        return result.get("findings", [])
