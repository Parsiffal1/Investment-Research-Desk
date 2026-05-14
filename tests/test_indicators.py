from investment_research_desk.providers.fixtures import FixtureProvider
from investment_research_desk.tools.indicators import atr, macd, max_drawdown, realized_volatility, rsi, support_resistance


def test_indicators_return_values_for_fixture():
    data = FixtureProvider().load("gold_cpi")

    assert rsi(data.ohlcv) is not None
    macd_line, signal, hist = macd(data.ohlcv)
    assert macd_line is not None
    assert signal is not None
    assert hist is not None
    assert atr(data.ohlcv) is not None
    assert realized_volatility(data.ohlcv) is not None
    assert max_drawdown(data.ohlcv) <= 0
    supports, resistances = support_resistance(data.ohlcv)
    assert supports
    assert resistances

