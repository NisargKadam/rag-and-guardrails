from functools import lru_cache

from fitness_agent.config import settings


@lru_cache
def get_chat_model():
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.ollama_model, temperature=0)

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model)

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (use 'openai' or 'ollama')")
