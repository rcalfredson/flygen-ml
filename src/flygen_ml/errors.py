from __future__ import annotations


class MalformedRecordingError(ValueError):
    """Raised when an upstream .data/.trx payload is structurally incomplete."""
