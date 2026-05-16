from investment_research_desk.tools.guardrails import find_guardrail_violations


def test_guardrail_allows_research_context_warning():
    text = "Use as research context only. This brief discusses market risk and scenario analysis."

    assert find_guardrail_violations(text) == []


def test_guardrail_allows_chinese_research_context_warning():
    text = "仅作为投研上下文使用。该报告讨论市场风险和情景分析。"

    assert find_guardrail_violations(text) == []


def test_guardrail_blocks_trading_advice_language():
    text = "Buy now, place a buy order, and use 20% of your portfolio. Guaranteed profit."

    violations = find_guardrail_violations(text)
    assert "direct_buy" in violations
    assert "position_sizing" in violations
    assert "guaranteed_profit" in violations
