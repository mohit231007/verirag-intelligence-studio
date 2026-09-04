import pytest

from core.config import AppConfig
from core.providers import GroqProvider, OllamaProvider, ProviderError, build_provider


def test_missing_groq_key_fails_without_network() -> None:
    provider = GroqProvider("", "model")
    with pytest.raises(ProviderError, match="not configured"):
        provider.complete("system", "user")


def test_build_provider_selects_runtime() -> None:
    assert isinstance(build_provider(AppConfig(provider="groq")), GroqProvider)
    assert isinstance(build_provider(AppConfig(provider="ollama")), OllamaProvider)
