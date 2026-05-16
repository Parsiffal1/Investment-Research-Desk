from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_root_readme_pair_has_language_switch_and_navigation():
    en = _read(REPO_ROOT / "README.md")
    zh = _read(REPO_ROOT / "README.zh.md")
    switch = "[English](README.md) | [中文](README.zh.md)"
    hero = "docs/assets/investment-research-desk-hero.gif"
    huashu = "https://github.com/alchaincyf/huashu-design/tree/master"
    assert switch in en
    assert switch in zh
    assert "docs/README.md" in en
    assert "docs/README.zh.md" in zh
    assert hero in en
    assert hero in zh
    assert huashu in en
    assert huashu in zh
    assert "docs/assets/screenshots/cli-interactive-menu.png" in en
    assert "docs/assets/screenshots/cli-live-progress.png" in zh
    assert "Investment Research Desk" in en
    assert "投研策略台" in zh
    assert not en.lstrip().startswith("# Investment Research Desk / 投研策略台")


def test_docs_index_pair_exists_and_links_to_core_docs():
    en = _read(DOCS_ROOT / "README.md")
    zh = _read(DOCS_ROOT / "README.zh.md")
    switch = "[English](README.md) | [中文](README.zh.md)"
    assert switch in en
    assert switch in zh
    for rel in [
        "current_implementation.md",
        "windows_cli_guide.md",
        "wsl_lora_adapter_guide.md",
        "lora_training_wsl.md",
    ]:
        assert rel in en
        assert rel in zh


def test_hero_assets_exist():
    assert (DOCS_ROOT / "assets" / "investment-research-desk-hero.html").exists()
    assert (DOCS_ROOT / "assets" / "investment-research-desk-hero.gif").exists()
    assert (REPO_ROOT / "scripts" / "render_true30_gif.cjs").exists()


def test_readme_uses_sanitized_env_example_values():
    en = _read(REPO_ROOT / "README.md")
    env_example = _read(REPO_ROOT / ".env.example")
    assert "your_tavily_api_key" in en
    assert "your_fmp_api_key" in en
    assert "your_finnhub_api_key" in en
    assert "your_jin10_api_key" in en
    assert "ghp_" not in env_example
    assert "your_tavily_api_key" in env_example
    assert "your_fmp_api_key" in env_example
    assert "your_finnhub_api_key" in env_example
