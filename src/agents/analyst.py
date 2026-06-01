from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

ANALYST_PROMPT = """You are a research analyst. Given search results, extract structured insights.

Output JSON:
{{
    "key_findings": ["finding 1", "finding 2"],
    "data_points": [{{"fact": "...", "source": "url"}}],
    "contradictions": ["any conflicting information"],
    "gaps": ["what's still unclear or missing"],
    "needs_more_research": true,
    "follow_up_queries": ["if needs_more_research, what to search next"]
}}

Rules:
- Be thorough in identifying key findings
- Note any contradictions between sources
- Identify clear gaps in the research
- Set needs_more_research to false if findings adequately answer the query
- Return ONLY valid JSON, no markdown fences"""


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
            f"Source: {r.get('url', 'N/A')}\nContent: {str(r.get('content', ''))[:1000]}"
            for r in search_results
        )
        return self.chain.invoke({
            "input": f"Query: {query}\n\nSearch Results:\n{formatted}"
        })
