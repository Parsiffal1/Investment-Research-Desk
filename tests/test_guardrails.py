from investment_research_desk.tools.guardrails import find_guardrail_violations


def test_guardrail_allows_research_context_warning():
    text = "Use as research context only. This brief discusses market risk and scenario analysis."

    assert find_guardrail_violations(text) == []


def test_guardrail_blocks_trading_advice_language():
    text = "Buy now, place a buy order, and use 20% of your portfolio. Guaranteed profit."

    violations = find_guardrail_violations(text)
    assert "direct_buy" in violations
    assert "position_sizing" in violations
    assert "guaranteed_profit" in violations

