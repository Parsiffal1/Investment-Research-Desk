from __future__ import annotations

from collections.abc import Iterable


def redact_secrets(text: str, secrets: Iterable[str | None]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "***")
    return redacted

