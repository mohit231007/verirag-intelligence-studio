import pytest

from core.config import load_config


def test_default_config_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERIRAG_PROVIDER", raising=False)
    monkeypatch.delenv("VERIRAG_MAX_FILE_MB", raising=False)
    config = load_config()
    assert config.max_file_bytes == 5 * 1024 * 1024
    assert 0.0 <= config.similarity_threshold <= 1.0
    assert config.chunk_overlap_chars < config.chunk_size_chars


def test_invalid_provider_fails_early(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIRAG_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="groq.*ollama"):
        load_config()


def test_overlap_must_be_smaller_than_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIRAG_CHUNK_SIZE_CHARS", "500")
    monkeypatch.setenv("VERIRAG_CHUNK_OVERLAP_CHARS", "500")
    with pytest.raises(ValueError, match="smaller"):
        load_config()
