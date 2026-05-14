from investment_research_desk.llm.clients import OllamaLLMClient


def test_ollama_json_repair_path():
    client = OllamaLLMClient("http://localhost:11434/v1", "qwen3:8b")
    calls = iter(["not json", '{"ok": true}'])
    client._chat_content = lambda system, user: next(calls)  # type: ignore[method-assign]

    assert client.chat_json("system", "user") == {"ok": True}

