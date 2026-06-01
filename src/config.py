from __future__ import annotations

import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


class Settings:
    def __init__(
        self,
        provider: str = "",
        llm_api_key: str = "",
        tavily_api_key: str = "",
        model: str = "",
    ):
        self.provider = LLMProvider(provider or os.getenv("LLM_PROVIDER", "gemini"))
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY", "")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", self._default_model())
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))

    def _default_model(self) -> str:
        if self.provider == LLMProvider.OPENAI:
            return "gpt-4o-mini"
        return "gemini-2.5-flash"

    def get_llm(self):
        if self.provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=self.model, api_key=self.llm_api_key, temperature=0)

        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.llm_api_key,
            temperature=0,
        )
