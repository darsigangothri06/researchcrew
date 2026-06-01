from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

PLANNER_PROMPT = """You are a research planner. Break the user's research query into \
specific, actionable subtasks.

Output JSON:
{{
    "subtasks": [
        {{"type": "search", "query": "specific search query", "priority": 1}},
        {{"type": "analyze", "query": "what to analyze from search results", "priority": 2}}
    ],
    "reasoning": "Why these subtasks cover the research question"
}}

Rules:
- 3-6 subtasks maximum
- "search" tasks find information; "analyze" tasks synthesize/compare findings
- Order by priority (1 = most important)
- Be specific in search queries — vague queries get poor results
- Return ONLY valid JSON, no markdown fences"""


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
            context = "\n\nFindings so far:\n" + "\n".join(existing_findings)
        result = self.chain.invoke({"input": f"Research query: {query}{context}"})
        subtasks = result.get("subtasks", [])
        for task in subtasks:
            task.setdefault("status", "pending")
        return subtasks
