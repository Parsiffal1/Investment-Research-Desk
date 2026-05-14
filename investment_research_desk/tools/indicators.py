from __future__ import annotations

import math
from statistics import mean, pstdev

from investment_research_desk.schemas import OHLCVBar


def _closes(bars: list[OHLCVBar]) -> list[float]:
    return [bar.close for bar in bars]


def sma(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    return mean(values[-period:])


def ema_series(values: list[float], period: int) -> list[float]:
    if not values or period <= 0:
        return []
    alpha = 2 / (period + 1)
    out = [values[0]]
    for value in values[1:]:
        out.append((value * alpha) + (out[-1] * (1 - alpha)))
    return out


def rsi(bars: list[OHLCVBar], period: int = 14) -> float | None:
    closes = _closes(bars)
    if len(closes) <= period:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for prev, curr in zip(closes[-period - 1 : -1], closes[-period:]):
        delta = curr - prev
        gains.append(max(delta, 0))
        losses.append(abs(min(delta, 0)))
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def macd(bars: list[OHLCVBar]) -> tuple[float | None, float | None, float | None]:
    closes = _closes(bars)
    if len(closes) < 26:
        return None, None, None
    fast = ema_series(closes, 12)
    slow = ema_series(closes, 26)
    line = [f - s for f, s in zip(fast[-len(slow) :], slow)]
    signal = ema_series(line, 9)
    if not line or not signal:
        return None, None, None
    hist = line[-1] - signal[-1]
    return round(line[-1], 4), round(signal[-1], 4), round(hist, 4)


def atr(bars: list[OHLCVBar], period: int = 14) -> float | None:
    if len(bars) <= period:
        return None
    true_ranges: list[float] = []
    recent = bars[-period:]
    previous = bars[-period - 1]
    for bar in recent:
        true_ranges.append(
            max(
                bar.high - bar.low,
                abs(bar.high - previous.close),
                abs(bar.low - previous.close),
            )
        )
        previous = bar
    return round(mean(true_ranges), 4)


def bollinger_state(bars: list[OHLCVBar], period: int = 20, width: float = 2.0) -> str:
    closes = _closes(bars)
    if len(closes) < period:
        return "insufficient_data"
    window = closes[-period:]
    mid = mean(window)
    std = pstdev(window)
    upper = mid + width * std
    lower = mid - width * std
    latest = closes[-1]
    if latest > upper:
        return "above_upper_band"
    if latest < lower:
        return "below_lower_band"
    if latest > mid:
        return "above_mid_band"
    return "below_mid_band"


def realized_volatility(bars: list[OHLCVBar]) -> float | None:
    closes = _closes(bars)
    if len(closes) < 3:
        return None
    returns = [math.log(curr / prev) for prev, curr in zip(closes, closes[1:]) if prev > 0]
    if len(returns) < 2:
        return None
    return round(pstdev(returns) * math.sqrt(252), 4)


def max_drawdown(bars: list[OHLCVBar]) -> float | None:
    closes = _closes(bars)
    if not closes:
        return None
    peak = closes[0]
    worst = 0.0
    for value in closes:
        peak = max(peak, value)
        if peak:
            worst = min(worst, (value - peak) / peak)
    return round(worst, 4)


def support_resistance(bars: list[OHLCVBar], levels: int = 2) -> tuple[list[float], list[float]]:
    if not bars:
        return [], []
    recent = bars[-20:]
    lows = sorted({round(bar.low, 2) for bar in recent})
    highs = sorted({round(bar.high, 2) for bar in recent}, reverse=True)
    return lows[:levels], highs[:levels]


def trend_label(bars: list[OHLCVBar]) -> str:
    closes = _closes(bars)
    if len(closes) < 20:
        return "insufficient_data"
    short = sma(closes, 5)
    long = sma(closes, 20)
    if short is None or long is None:
        return "insufficient_data"
    if short > long and closes[-1] > closes[-5]:
        return "uptrend"
    if short < long and closes[-1] < closes[-5]:
        return "downtrend"
    return "range_bound"

