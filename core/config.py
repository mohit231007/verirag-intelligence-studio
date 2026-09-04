"""Typed, environment-driven application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _as_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, received {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _as_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, received {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime settings with conservative public-demo limits."""

    app_name: str = "VeriRAG Studio"
    provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.40
    top_k: int = 4
    candidate_multiplier: int = 3
    chunk_size_chars: int = 1_800
    chunk_overlap_chars: int = 220
    max_file_bytes: int = 5 * 1024 * 1024
    max_files: int = 5
    max_pages_per_file: int = 200
    max_chunks_per_session: int = 1_500
    max_context_chars: int = 16_000
    request_timeout_seconds: int = 60

    @property
    def provider_ready(self) -> bool:
        return self.provider == "ollama" or bool(self.groq_api_key)


def load_config() -> AppConfig:
    """Load configuration from the environment and fail early on unsafe values."""

    provider = os.getenv("VERIRAG_PROVIDER", "groq").strip().lower()
    if provider not in {"groq", "ollama"}:
        raise ValueError("VERIRAG_PROVIDER must be either 'groq' or 'ollama'")

    chunk_size = _as_int("VERIRAG_CHUNK_SIZE_CHARS", 1_800, 400, 8_000)
    overlap = _as_int("VERIRAG_CHUNK_OVERLAP_CHARS", 220, 0, 2_000)
    if overlap >= chunk_size:
        raise ValueError("VERIRAG_CHUNK_OVERLAP_CHARS must be smaller than chunk size")

    return AppConfig(
        provider=provider,
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip(),
        embedding_model=os.getenv(
            "VERIRAG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip(),
        similarity_threshold=_as_float("VERIRAG_SIMILARITY_THRESHOLD", 0.40, 0.0, 1.0),
        top_k=_as_int("VERIRAG_TOP_K", 4, 1, 12),
        candidate_multiplier=_as_int("VERIRAG_CANDIDATE_MULTIPLIER", 3, 1, 10),
        chunk_size_chars=chunk_size,
        chunk_overlap_chars=overlap,
        max_file_bytes=_as_int("VERIRAG_MAX_FILE_MB", 5, 1, 25) * 1024 * 1024,
        max_files=_as_int("VERIRAG_MAX_FILES", 5, 1, 20),
        max_pages_per_file=_as_int("VERIRAG_MAX_PAGES", 200, 1, 1_000),
        max_chunks_per_session=_as_int("VERIRAG_MAX_CHUNKS", 1_500, 10, 10_000),
        max_context_chars=_as_int("VERIRAG_MAX_CONTEXT_CHARS", 16_000, 2_000, 100_000),
        request_timeout_seconds=_as_int("VERIRAG_REQUEST_TIMEOUT", 60, 5, 300),
    )
