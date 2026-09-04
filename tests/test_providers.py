import pytest

from core.config import AppConfig
from core.providers import GroqProvider, OllamaProvider, ProviderError, build_provider, groq_error_message


def test_missing_groq_key_fails_without_network() -> None:
    provider = GroqProvider("", "model")
    with pytest.raises(ProviderError, match="not configured"):
        provider.complete("system", "user")


def test_build_provider_selects_runtime() -> None:
    assert isinstance(build_provider(AppConfig(provider="groq")), GroqProvider)
    assert isinstance(build_provider(AppConfig(provider="ollama")), OllamaProvider)


@pytest.mark.parametrize(
    ("status", "expected"),
    [(400, "model"), (401, "API_KEY"), (403, "access"), (404, "unavailable"), (429, "limit")],
)
def test_groq_errors_are_actionable_without_response_body(status, expected) -> None:
    error = type("HttpError", (Exception,), {"status_code": status})("sensitive upstream response")
    message = groq_error_message(error)
    assert expected in message
    assert "sensitive" not in message
