from investment_research_desk.agents import contract_manifest, get_agent_contract
from investment_research_desk.graph import ResearchWorkflow
from investment_research_desk.providers.fixtures import FixtureProvider


EXPECTED_CONTRACTS = {
    "run_controller",
    "data_ingestion",
    "fundamental_macro",
    "news_impact",
    "sentiment",
    "technical",
    "analyst_team",
    "bull_researcher",
    "bear_researcher",
    "bull_bear_research_debate",
    "research_reporter",
    "final_market_context_cache",
    "persist",
}


def test_agent_contract_manifest_covers_workflow_nodes():
    manifest = contract_manifest()

    assert EXPECTED_CONTRACTS.issubset(manifest)
    for name in EXPECTED_CONTRACTS:
        contract = get_agent_contract(name)
        assert contract.role
        assert contract.allowed_inputs
        assert contract.allowed_tools
        assert contract.output_schema
        assert contract.system_prompt
        assert any("position sizing" in action for action in contract.forbidden_actions)
        assert any("buy" in action and "sell" in action for action in contract.forbidden_actions)


def test_agent_data_scopes_do_not_expose_unrelated_inputs():
    data = FixtureProvider().load("gold_cpi")

    technical = ResearchWorkflow._scope_data(data, "technical")
    assert technical.ohlcv
    assert technical.news_events == []
    assert technical.sentiment_inputs == []
    assert technical.source_metadata == {}

    sentiment = ResearchWorkflow._scope_data(data, "sentiment")
    assert sentiment.sentiment_inputs
    assert sentiment.ohlcv == []
    assert sentiment.news_events == []

    news = ResearchWorkflow._scope_data(data, "news_impact")
    assert news.news_events
    assert news.ohlcv == []
    assert news.sentiment_inputs == []

