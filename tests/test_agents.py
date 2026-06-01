from src.config import Settings, LLMProvider


def test_settings_defaults():
    s = Settings(provider="gemini", llm_api_key="test-key")
    assert s.provider == LLMProvider.GEMINI
    assert s.model == "gemini-2.5-flash"
    assert s.llm_api_key == "test-key"


def test_settings_openai():
    s = Settings(provider="openai", llm_api_key="test-key")
    assert s.provider == LLMProvider.OPENAI
    assert s.model == "gpt-4o-mini"


def test_settings_custom_model():
    s = Settings(provider="gemini", llm_api_key="k", model="gemini-1.5-pro")
    assert s.model == "gemini-1.5-pro"
