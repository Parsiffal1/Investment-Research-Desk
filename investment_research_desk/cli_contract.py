from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from investment_research_desk.providers.fixtures import FixtureProvider
from investment_research_desk.schemas import RunRequest


class AssetClassOption(str, Enum):
    crypto = "crypto"
    precious_metal = "precious_metal"
    equity_index = "equity_index"
    commodity = "commodity"
    fx = "fx"
    equity = "equity"
    other = "other"


class HorizonOption(str, Enum):
    intraday = "intraday"
    short_term = "short_term"
    swing = "swing"
    medium_term = "medium_term"


class ResearchDepthOption(str, Enum):
    quick = "quick"
    standard = "standard"
    deep = "deep"


class LLMProviderOption(str, Enum):
    auto = "auto"
    fake = "fake"
    ollama = "ollama"


ALLOWED_ASSET_CLASSES = tuple(item.value for item in AssetClassOption)
ALLOWED_HORIZONS = tuple(item.value for item in HorizonOption)
ALLOWED_RESEARCH_DEPTHS = tuple(item.value for item in ResearchDepthOption)
ALLOWED_LLM_PROVIDERS = tuple(item.value for item in LLMProviderOption)

REQUIRED_ARTIFACTS = (
    "input.json",
    "agent_contracts.json",
    "normalized_data.json",
    "analyst_outputs.json",
    "analyst_team_outputs.json",
    "bull_risk_outputs.json",
    "research_debate.json",
    "final_market_context_cache.json",
    "final_research_context.json",
    "research_brief.md",
    "trace.json",
    "metrics.json",
)

TEAM_FLOW = (
    ("Run Control", ("run_controller",)),
    (
        "Analyst Team",
        ("fundamental_macro", "news_impact", "sentiment", "technical"),
    ),
    ("Bull/Bear Research Debate", ("bull_researcher", "bear_researcher", "bull_bear_research_debate")),
    ("Research Reporter", ("research_reporter", "final_market_context_cache", "persist")),
)


@dataclass(frozen=True)
class CLIInteractionContract:
    mode: str
    request: RunRequest | None
    checkpoint: bool
    resume_run_id: str | None
    runs_dir: Path | None
    clear_checkpoints: bool = False


def allowed_text(values: Iterable[str]) -> str:
    return ", ".join(values)


def normalize_enum_value(value: str | Enum, allowed: tuple[str, ...], field_name: str) -> str:
    raw = value.value if isinstance(value, Enum) else value
    normalized = str(raw).strip()
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of: {allowed_text(allowed)}")
    return normalized


def build_run_request(
    *,
    symbol: str | None,
    asset_class: str | AssetClassOption,
    horizon: str | HorizonOption,
    research_depth: str | ResearchDepthOption,
    fixture: str | None,
    llm_provider: str | LLMProviderOption,
    model: str | None,
) -> RunRequest:
    provider = normalize_enum_value(llm_provider, ALLOWED_LLM_PROVIDERS, "llm_provider")
    depth = normalize_enum_value(research_depth, ALLOWED_RESEARCH_DEPTHS, "research_depth")

    try:
        if fixture:
            request = FixtureProvider().request(fixture)
            request.llm_provider = provider  # type: ignore[assignment]
            request.model = model
            request.research_depth = depth  # type: ignore[assignment]
            return request

        normalized_symbol = (symbol or "").strip().upper()
        if not normalized_symbol:
            raise ValueError("--symbol is required when --fixture is not used")

        return RunRequest(
            symbol=normalized_symbol,
            asset_class=normalize_enum_value(asset_class, ALLOWED_ASSET_CLASSES, "asset_class"),  # type: ignore[arg-type]
            horizon=normalize_enum_value(horizon, ALLOWED_HORIZONS, "horizon"),  # type: ignore[arg-type]
            research_depth=depth,  # type: ignore[arg-type]
            llm_provider=provider,  # type: ignore[arg-type]
            model=model.strip() if model and model.strip() else None,
        )
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def discover_fixtures(fixtures_dir: Path = Path("data/fixtures")) -> list[str]:
    if not fixtures_dir.exists():
        return []
    return sorted(path.stem for path in fixtures_dir.glob("*.json"))


def discover_runs(runs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not runs_dir.exists():
        return rows
    for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), reverse=True):
        checkpoint = run_dir / "checkpoint.json"
        final_context = run_dir / "final_research_context.json"
        rows.append(
            {
                "run_id": run_dir.name,
                "status": "checkpoint" if checkpoint.exists() else ("complete" if final_context.exists() else "partial"),
                "checkpoint": str(checkpoint) if checkpoint.exists() else "",
                "final_context": str(final_context) if final_context.exists() else "",
            }
        )
    return rows
