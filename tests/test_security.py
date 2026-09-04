import pytest

from core.security import UploadValidationError, sanitize_filename, validate_upload


def test_filename_drops_paths_and_unsafe_characters() -> None:
    assert sanitize_filename("../../Q4<policy>.txt") == "Q4_policy_.txt"


def test_upload_rejects_disallowed_type() -> None:
    with pytest.raises(UploadValidationError, match="Unsupported"):
        validate_upload("payload.exe", b"not really an exe", 100)


def test_upload_rejects_binary_text() -> None:
    with pytest.raises(UploadValidationError, match="binary"):
        validate_upload("notes.txt", b"hello\x00world", 100)


def test_upload_rejects_oversize() -> None:
    with pytest.raises(UploadValidationError, match="limit"):
        validate_upload("notes.txt", b"1234", 3)
