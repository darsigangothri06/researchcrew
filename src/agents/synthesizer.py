from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYNTHESIZER_PROMPT = """You are a research report writer. Synthesize all findings into \
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

    def synthesize(
        self,
        query: str,
        findings: list[dict],
        notes: list[str],
        sources: list[dict],
    ) -> str:
        findings_text = "\n\n".join(str(f) for f in findings)
        notes_text = "\n".join(notes) if notes else "No additional notes."
        sources_text = "\n".join(
            f"- {s.get('url', 'N/A')}: {s.get('title', '')}" for s in sources
        ) or "No sources collected."

        return self.chain.invoke({
            "input": (
                f"Query: {query}\n\n"
                f"Findings:\n{findings_text}\n\n"
                f"Notes:\n{notes_text}\n\n"
                f"Sources:\n{sources_text}"
            )
        })
