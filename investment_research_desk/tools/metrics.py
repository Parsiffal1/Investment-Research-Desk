from __future__ import annotations

import json
from typing import Any


def approximate_tokens(value: Any) -> int:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    if not value:
        return 0
    return max(1, int(len(value.split()) * 1.3))


def compression_ratio(raw_tokens: int, final_tokens: int) -> float:
    if final_tokens <= 0:
        return 0.0
    return round(raw_tokens / final_tokens, 2)

