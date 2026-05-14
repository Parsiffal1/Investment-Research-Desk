from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Protocol

import httpx

from investment_research_desk.config import Settings


class LLMClient(Protocol):
    provider: str
    model: str

    def chat_json(self, system: str, user: str) -> dict:
        ...

    def healthcheck(self) -> tuple[bool, str]:
        ...


class FakeLLMClient:
    provider = "fake"
    model = "deterministic-fake-llm"

    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def chat_json(self, system: str, user: str) -> dict:
        self.calls.append({"system": system, "user": user})
        candidate = _candidate_from_prompt(user)
        if candidate is not None:
            return candidate
        lowered = f"{system}\n{user}".lower()
        if "sentiment" in lowered:
            return {"label": "mixed", "score": 0.0, "summary": "Fixture-backed mixed sentiment."}
        if "news" in lowered or "impact" in lowered:
            return {"impact": "mixed", "summary": "Macro events create mixed market impact."}
        return {"summary": "Deterministic fixture-backed research output."}

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake LLM is available"


class OllamaLLMClient:
    provider = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat_json(self, system: str, user: str) -> dict:
        content = self._chat_content(system, user)
        try:
            return _parse_json_object(content)
        except (JSONDecodeError, ValueError):
            repair_system = "Return only one valid JSON object. Do not include markdown or commentary."
            repair_user = f"Repair this invalid JSON-like output into valid JSON:\n{content}"
            return _parse_json_object(self._chat_content(repair_system, repair_user))

    def _chat_content(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def healthcheck(self) -> tuple[bool, str]:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/models")
            if response.status_code >= 400:
                return False, f"Ollama endpoint returned HTTP {response.status_code}"
            models = response.json().get("data", [])
            names = {item.get("id") for item in models}
            if names and self.model not in names:
                return False, f"Ollama reachable, but model '{self.model}' was not listed"
            return True, "Ollama endpoint is reachable"
        except Exception as exc:
            return False, f"Ollama unavailable: {exc}"


def make_llm_client(settings: Settings, provider: str, model: str | None = None, allow_fake_fallback: bool = False) -> LLMClient:
    if provider == "fake":
        return FakeLLMClient()
    if provider in {"ollama", "auto"}:
        ollama = OllamaLLMClient(settings.ollama_base_url, model or settings.ollama_model)
        ok, _ = ollama.healthcheck()
        if ok or provider == "ollama":
            return ollama
        if allow_fake_fallback:
            return FakeLLMClient()
        return ollama
    raise ValueError(f"Unsupported llm provider: {provider}")


def _parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _candidate_from_prompt(user: str) -> dict | None:
    marker = "Candidate output JSON:"
    start = user.find(marker)
    if start < 0:
        return None
    text = user[start + len(marker) :].strip()
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
