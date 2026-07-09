"""Frontier-model evaluators that score a captured run's output.

Reads provider keys from ``.env`` in the working directory:

  - ``GEMINI_API_KEY``    — enables Google Gemini models (model id starts with ``gemini``)
  - ``ANTHROPIC_API_KEY`` — enables Anthropic Claude models (model id starts with ``claude``)
  - ``OPENAI_API_KEY``    — enables OpenAI GPT models (model id starts with ``gpt`` or ``o1``)

Provider is inferred from the configured model id. Network calls go through
``urllib`` so no extra runtime dependency is required.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .scenario import LLMValidateConfig, Message


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Populate ``os.environ`` from a ``.env`` file (no-op if missing).

    Returns the keys that were loaded so callers can report what is available.
    Existing environment variables are not overwritten.
    """
    env_path = path if path is not None else Path.cwd() / ".env"
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def available_providers() -> dict[str, bool]:
    load_dotenv()
    return {
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
        "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
    }


def _provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("claude"):
        return "claude"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    raise ValueError(
        f"Cannot infer provider from model id {model!r}. "
        "Use a name starting with gemini-, claude-, gpt-, or o1/o3/o4-."
    )


def render_template(
    template: str,
    *,
    test_system_prompt: str,
    test_prompt: str,
    model_output: str,
    full_conversation: str,
) -> str:
    return (
        template.replace("{{test_system_prompt}}", test_system_prompt)
        .replace("{{test_prompt}}", test_prompt)
        .replace("{{model_output}}", model_output)
        .replace("{{full_conversation}}", full_conversation)
    )


def format_conversation(messages: list[Message], generated_text: str) -> str:
    lines = [f"{m.role}: {m.content}" for m in messages]
    if generated_text:
        lines.append(f"assistant: {generated_text}")
    return "\n\n".join(lines)


def extract_test_inputs(
    messages: list[Message], generated_text: str
) -> tuple[str, str, str]:
    """Return (test_system_prompt, test_prompt, model_output) from a run."""
    system_prompt = next((m.content for m in messages if m.role == "system"), "")
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )
    return system_prompt, last_user, generated_text


def _http_post_json(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{e.code} {e.reason}: {detail}") from e


def _http_get_json(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{e.code} {e.reason}: {detail}") from e


def _list_gemini_models() -> list[str]:
    api_key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    data = _http_get_json(url, {})
    skip_substrings = ("tts", "image", "audio", "embedding", "computer-use", "robotics")
    models: list[str] = []
    for entry in data.get("models", []):
        name = entry.get("name", "")
        methods = entry.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        # Names come back as "models/gemini-1.5-pro"; strip the prefix to match the API call format.
        short = name.split("/", 1)[1] if name.startswith("models/") else name
        # Only keep canonical gemini-* chat models — provider routing keys off the prefix.
        if not short.startswith("gemini-"):
            continue
        if any(s in short for s in skip_substrings):
            continue
        models.append(short)
    return sorted(models)


def _list_claude_models() -> list[str]:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    data = _http_get_json("https://api.anthropic.com/v1/models?limit=1000", headers)
    return sorted(entry["id"] for entry in data.get("data", []) if entry.get("id"))


def _list_openai_models() -> list[str]:
    api_key = os.environ["OPENAI_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}"}
    data = _http_get_json("https://api.openai.com/v1/models", headers)
    chat_prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt-")
    skip_substrings = ("audio", "realtime", "transcribe", "tts", "image", "embedding", "whisper", "dall-e", "moderation")
    out: list[str] = []
    for entry in data.get("data", []):
        mid = entry.get("id", "")
        low = mid.lower()
        if not any(low.startswith(p) for p in chat_prefixes):
            continue
        if any(s in low for s in skip_substrings):
            continue
        out.append(mid)
    return sorted(out)


def list_provider_models() -> dict[str, list[str] | dict[str, str]]:
    """Query each provider with a configured key and return its available chat models.

    Result shape:
        {provider: [model_id, ...] OR {"error": "..."}}

    Providers without a configured key are omitted entirely.
    """
    load_dotenv()
    available = available_providers()
    fetchers = {
        "gemini": _list_gemini_models,
        "claude": _list_claude_models,
        "openai": _list_openai_models,
    }
    out: dict[str, list[str] | dict[str, str]] = {}
    for provider, has_key in available.items():
        if not has_key:
            continue
        try:
            out[provider] = fetchers[provider]()
        except Exception as exc:
            out[provider] = {"error": str(exc)}
    return out


def _call_gemini(model: str, system_prompt: str, prompt: str) -> str:
    api_key = os.environ["GEMINI_API_KEY"]
    if not api_key:
        raise Exception("GEMINI API KEY environment variable is missing")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    data = _http_post_json(url, payload, {"Content-Type": "application/json"})
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _call_claude(model: str, system_prompt: str, prompt: str) -> str:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    url = "https://api.anthropic.com/v1/messages"
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        payload["system"] = system_prompt
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    data = _http_post_json(url, payload, headers)
    blocks = data.get("content") or []
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _call_openai(model: str, system_prompt: str, prompt: str) -> str:
    api_key = os.environ["OPENAI_API_KEY"]
    url = "https://api.openai.com/v1/chat/completions"
    msgs: list[dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": msgs}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = _http_post_json(url, payload, headers)
    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""


def run_validator(
    cfg: LLMValidateConfig,
    *,
    test_system_prompt: str,
    test_prompt: str,
    model_output: str,
    full_conversation: str,
) -> dict[str, Any]:
    """Render templates, call the inferred provider, return a result dict."""
    provider = _provider_for(cfg.model)
    available = available_providers()
    if not available.get(provider):
        env_var = {
            "gemini": "GEMINI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }[provider]
        raise RuntimeError(
            f"Provider {provider!r} required by model {cfg.model!r} but {env_var} is not set."
        )

    rendered_system = render_template(
        cfg.system_prompt,
        test_system_prompt=test_system_prompt,
        test_prompt=test_prompt,
        model_output=model_output,
        full_conversation=full_conversation,
    )
    rendered_prompt = render_template(
        cfg.prompt,
        test_system_prompt=test_system_prompt,
        test_prompt=test_prompt,
        model_output=model_output,
        full_conversation=full_conversation,
    )

    caller = {"gemini": _call_gemini, "claude": _call_claude, "openai": _call_openai}[provider]
    output = caller(cfg.model, rendered_system, rendered_prompt)

    return {
        "model": cfg.model,
        "provider": provider,
        "system_prompt_template": cfg.system_prompt,
        "prompt_template": cfg.prompt,
        "rendered_system_prompt": rendered_system,
        "rendered_prompt": rendered_prompt,
        "output": output,
    }
