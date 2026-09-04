"""Minimal provider adapters for free cloud and local inference."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .config import AppConfig


class ProviderError(RuntimeError):
    """A sanitized error safe to display in the application."""


def groq_error_message(error: Exception) -> str:
    """Translate Groq HTTP failures without exposing credentials or response bodies."""

    status = getattr(error, "status_code", None)
    if status == 400:
        return "Groq rejected the request. Check that the configured model is active for your account."
    if status == 401:
        return "Groq authentication failed. Check that GROQ_API_KEY is complete, active, and saved in .env."
    if status == 403:
        return "Groq denied access to this model. Select a model enabled for your Groq project."
    if status == 404:
        return "The configured Groq model is unavailable. Update GROQ_MODEL to an active model ID."
    if status == 429:
        return "Groq rate or usage limit reached. Wait briefly, then retry or review your Groq project limits."
    return "Groq request failed. Check network access and your Groq project status."


class ChatProvider(Protocol):
    name: str
    model: str

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str: ...


@dataclass(slots=True)
class GroqProvider:
    api_key: str
    model: str
    timeout_seconds: int = 60
    name: str = "groq"

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        if not self.api_key:
            raise ProviderError("GROQ_API_KEY is not configured")
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if not content:
                raise ProviderError("The model returned an empty response")
            return content.strip()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(groq_error_message(exc)) from exc


@dataclass(slots=True)
class OllamaProvider:
    base_url: str
    model: str
    timeout_seconds: int = 60
    name: str = "ollama"

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"temperature": temperature},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "").strip()
            if not content:
                raise ProviderError("Ollama returned an empty response")
            return content
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and pull {self.model}."
            ) from exc


def build_provider(config: AppConfig) -> ChatProvider:
    if config.provider == "ollama":
        return OllamaProvider(config.ollama_base_url, config.ollama_model, config.request_timeout_seconds)
    return GroqProvider(config.groq_api_key, config.groq_model, config.request_timeout_seconds)
