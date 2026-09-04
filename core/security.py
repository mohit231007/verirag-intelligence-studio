"""Input-validation helpers for an untrusted public upload surface."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePath

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")


class UploadValidationError(ValueError):
    """Raised when an uploaded document violates a public-demo boundary."""


def sanitize_filename(filename: str) -> str:
    """Drop path components and normalize unsafe display characters."""

    name = PurePath(filename.replace("\\", "/")).name.strip()
    name = _SAFE_FILENAME.sub("_", name)
    return name[:160] or "document"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_upload(filename: str, content: bytes, max_file_bytes: int) -> str:
    """Validate size and allowlisted extension; return the sanitized name."""

    safe_name = sanitize_filename(filename)
    extension = PurePath(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UploadValidationError(f"Unsupported file type {extension or '(none)'}. Allowed: {allowed}")
    if not content:
        raise UploadValidationError("The uploaded file is empty")
    if len(content) > max_file_bytes:
        limit_mb = max_file_bytes / (1024 * 1024)
        raise UploadValidationError(f"File exceeds the {limit_mb:g} MB limit")
    if b"\x00" in content[:8_192] and extension in {".txt", ".md", ".csv"}:
        raise UploadValidationError("Text-like file contains binary data")
    return safe_name


def neutralize_spreadsheet_formula(value: str) -> str:
    """Prevent formula execution if extracted CSV text is later exported."""

    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value
