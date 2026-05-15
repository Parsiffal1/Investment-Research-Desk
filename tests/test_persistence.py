import json
from pathlib import Path

from investment_research_desk.persistence import RunStore


def test_json_artifacts_are_ascii_escaped_for_powershell_compatibility(tmp_path: Path):
    store = RunStore(tmp_path)

    path = store.write_json(
        "run",
        "normalized_data.json",
        {
            "title": "投研策略台",
            "provider_text": "StockTwits: $SPY question is… What’s next?",
        },
    )

    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["title"] == "投研策略台"
    assert "\\u6295\\u7814\\u7b56\\u7565\\u53f0" in text
    assert "…" not in text
    assert "’" not in text
