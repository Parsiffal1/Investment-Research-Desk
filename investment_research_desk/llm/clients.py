from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Callable, Protocol

import httpx

from investment_research_desk.config import Settings


class LLMClient(Protocol):
    provider: str
    model: str

    def chat_json(self, system: str, user: str) -> dict:
        ...

    def chat_json_options(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> dict:
        ...

    def chat_tools_json(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        execute_tool: Callable[[str, dict[str, Any]], Any],
        max_rounds: int = 4,
    ) -> dict:
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
        try:
            payload = json.loads(user)
            if isinstance(payload, dict) and isinstance(payload.get("items"), list):
                allowed = payload.get("allowed_labels") or ["neutral"]
                label = allowed[0] if isinstance(allowed, list) and allowed else "neutral"
                return {"predictions": [{"id": item.get("id"), "label": label} for item in payload["items"]]}
        except json.JSONDecodeError:
            pass
        lowered = f"{system}\n{user}".lower()
        if "sentiment" in lowered:
            return {"label": "mixed", "score": 0.0, "summary": "Fixture-backed mixed sentiment."}
        if "news" in lowered or "impact" in lowered:
            return {"impact": "mixed", "summary": "Macro events create mixed market impact."}
        return {"summary": "Deterministic fixture-backed research output."}

    def chat_json_options(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> dict:
        return self.chat_json(system, user)

    def chat_tools_json(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        execute_tool: Callable[[str, dict[str, Any]], Any],
        max_rounds: int = 4,
    ) -> dict:
        self.calls.append({"system": system, "user": user})
        tool_calls = []
        for tool in tools[:2]:
            name = tool.get("function", {}).get("name")
            if not name:
                continue
            arguments = {"query": "fixture market news", "limit": 5}
            result = execute_tool(name, arguments)
            tool_calls.append({"name": name, "arguments": arguments, "result": result})
        candidate = _candidate_from_prompt(user) or {"summary": "Deterministic tool-backed output."}
        return {"result": candidate, "_tool_calls": tool_calls}

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake LLM is available"


class OllamaLLMClient:
    provider = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat_json(self, system: str, user: str) -> dict:
        return self.chat_json_options(system, user, temperature=0.1)

    def chat_json_options(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> dict:
        content = self._chat_content(system, user, temperature=temperature, max_tokens=max_tokens, seed=seed)
        try:
            return _parse_json_object(content)
        except (JSONDecodeError, ValueError):
            repair_system = "Return only one valid JSON object. Do not include markdown or commentary."
            repair_user = f"Repair this invalid JSON-like output into valid JSON:\n{content}"
            return _parse_json_object(
                self._chat_content(repair_system, repair_user, temperature=0.0, max_tokens=max_tokens, seed=seed)
            )

    def chat_tools_json(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        execute_tool: Callable[[str, dict[str, Any]], Any],
        max_rounds: int = 4,
    ) -> dict:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tool_calls_log: list[dict[str, Any]] = []

        for _ in range(max_rounds):
            message = self._chat_message(messages, tools=tools, response_format=None)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = message.get("content") or "{}"
                try:
                    parsed = _parse_json_object(content)
                except (JSONDecodeError, ValueError):
                    parsed = self.chat_json(
                        "Return only one valid JSON object. Do not include markdown or commentary.",
                        f"Repair this final tool-loop output into valid JSON:\n{content}",
                    )
                parsed["_tool_calls"] = tool_calls_log
                return parsed

            assistant_message = {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            }
            messages.append(assistant_message)
            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name")
                if not name:
                    continue
                arguments = _parse_tool_arguments(function.get("arguments"))
                result = execute_tool(name, arguments)
                tool_call_id = call.get("id") or f"call_{len(tool_calls_log)}"
                tool_calls_log.append({"name": name, "arguments": arguments, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        final_user = (
            "Stop calling tools now. Using only the tool results already provided, "
            "return the final JSON object matching the requested schema."
        )
        messages.append({"role": "user", "content": final_user})
        message = self._chat_message(messages, tools=[], response_format={"type": "json_object"})
        parsed = _parse_json_object(message.get("content") or "{}")
        parsed["_tool_calls"] = tool_calls_log
        return parsed

    def _chat_content(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        return self._chat_message(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )["content"]

    def _chat_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_format: dict[str, str] | None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
        return response.json()["choices"][0]["message"]

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


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


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
