from __future__ import annotations

import re


PROHIBITED_PATTERNS = {
    "direct_buy": re.compile(r"\b(buy now|must buy|strong buy immediately|place a buy order)\b", re.I),
    "direct_sell": re.compile(r"\b(sell now|must sell|place a sell order)\b", re.I),
    "position_sizing": re.compile(
        r"\b(use|allocate|risk)\s+\d+(\.\d+)?%\s+(of\s+your|of|your)\s+(portfolio|capital|account)\b",
        re.I,
    ),
    "guaranteed_profit": re.compile(r"\b(guaranteed profit|risk-free profit|cannot lose|sure win)\b", re.I),
    "order_instruction": re.compile(r"\b(limit order|market order|stop loss at|take profit at)\b", re.I),
}


REQUIRED_WARNING = "Use as research context only"


def find_guardrail_violations(text: str) -> list[str]:
    violations = [name for name, pattern in PROHIBITED_PATTERNS.items() if pattern.search(text)]
    if REQUIRED_WARNING.lower() not in text.lower():
        violations.append("missing_downstream_usage_warning")
    return violations
