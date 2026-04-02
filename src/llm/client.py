"""Unified LLM client with multi-provider dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from src.config import Config


class LLMClient:
    """Calls LLM APIs with provider-based dispatch.

    Usage:
        client = LLMClient(config)
        response = await client.complete("gemini", system_prompt, user_prompt)
    """

    def __init__(self, config: Config, timeout: float = 300.0):
        self._config = config
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def complete(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Send a completion request to the specified provider."""
        cfg = self._config.get_provider_config(provider)
        if not cfg:
            raise ValueError(f"Unknown LLM provider: {provider}")

        api_key = cfg.get("api_key", "")
        model = cfg.get("model", "")
        base_url = cfg.get("base_url", "")

        if provider == "gemini":
            return await self._call_gemini(base_url, api_key, model, system_prompt, user_prompt, temperature, max_tokens)
        elif provider == "claude":
            return await self._call_claude(base_url, api_key, model, system_prompt, user_prompt, temperature, max_tokens)
        else:
            # openai, deepseek 等所有 OpenAI 兼容接口走同一路径
            return await self._call_openai(base_url, api_key, model, system_prompt, user_prompt, temperature, max_tokens)

    async def _call_gemini(
        self, base_url: str, api_key: str, model: str,
        system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int,
    ) -> str:
        url = f"{base_url}/models/{model}:generateContent?key={api_key}"
        body: dict[str, Any] = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        resp = await self._http.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_claude(
        self, base_url: str, api_key: str, model: str,
        system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int,
    ) -> str:
        url = f"{base_url}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt

        resp = await self._http.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    async def _call_openai(
        self, base_url: str, api_key: str, model: str,
        system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int,
    ) -> str:
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = await self._http.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        # DeepSeek Reasoner: 实际回答在 content，推理过程在 reasoning_content
        # 如果 content 为空但有 reasoning_content，回退用 reasoning_content
        if not content and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return content


def load_prompt_template(template_path: str | Path) -> str:
    """Load a prompt template from a Markdown file."""
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
