from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchSettings(BaseModel):
    provider: str = Field(default="gemini", description="LLM provider: 'openai' or 'gemini'")
    llm_api_key: str = Field(default="", description="API key for the LLM provider")
    tavily_api_key: str = Field(default="", description="Tavily API key for web search")
    model: str = Field(default="", description="Model name override")


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Research topic or question")
    settings: ResearchSettings = Field(default_factory=ResearchSettings)


class StepEvent(BaseModel):
    event: str = "step"
    agent: str = ""
    step: str = ""
    content: str = ""


class CompleteEvent(BaseModel):
    event: str = "complete"
    report: str = ""
    sources: list[dict] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    report: str
    sources: list[dict] = Field(default_factory=list)
    iterations: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
