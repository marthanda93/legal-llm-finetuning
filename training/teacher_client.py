"""Pluggable client for the 'teacher' LLM used in Stage C distillation.

Supports:
    - Anthropic Claude (claude-sonnet-4-5 by default)         via ANTHROPIC_API_KEY
    - OpenAI GPT-4.1 / GPT-4o                                 via OPENAI_API_KEY
    - Google Gemini 2.5 Pro                                   via GOOGLE_API_KEY

Pick the one whose key is set; explicit `--provider` override always wins.

Usage:
    from training.teacher_client import get_teacher_client
    client = get_teacher_client()
    response_text = client.complete(system_prompt, user_prompt, max_tokens=2000)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import httpx  # type: ignore
except ImportError:  # noqa: F401
    httpx = None  # type: ignore


@dataclass
class TeacherClient:
    provider: str       # "anthropic" | "openai" | "gemini"
    model: str
    api_key: str
    base_url: str
    timeout: float = 120.0
    max_retries: int = 4

    def complete(self, system: str, user: str, max_tokens: int = 2000,
                 temperature: float = 0.4) -> str:
        if httpx is None:
            raise RuntimeError("httpx is required for teacher_client; pip install httpx")

        attempts = 0
        last_err: Optional[Exception] = None
        while attempts <= self.max_retries:
            try:
                if self.provider == "anthropic":
                    return self._anthropic(system, user, max_tokens, temperature)
                if self.provider == "openai":
                    return self._openai(system, user, max_tokens, temperature)
                if self.provider == "gemini":
                    return self._gemini(system, user, max_tokens, temperature)
                raise ValueError(f"unknown provider {self.provider}")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                attempts += 1
                # Exponential backoff for 429 / 5xx
                delay = min(60.0, 2.0 ** attempts)
                print(f"[teacher] attempt {attempts} failed ({exc}); sleeping {delay}s")
                time.sleep(delay)
        raise RuntimeError(f"teacher request failed after {self.max_retries} retries: {last_err}")

    # --- Anthropic ---
    def _anthropic(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        resp = httpx.post(self.base_url, json=body, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")

    # --- OpenAI / OpenAI-compat ---
    def _openai(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(self.base_url, json=body, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # --- Gemini ---
    def _gemini(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        url = f"{self.base_url}?key={self.api_key}"
        resp = httpx.post(url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)


def get_teacher_client(provider: Optional[str] = None) -> TeacherClient:
    """Auto-detect a teacher provider from env vars; allow explicit override."""

    chosen = (provider or os.getenv("TEACHER_PROVIDER") or "").strip().lower() or None

    if chosen is None:
        if os.getenv("ANTHROPIC_API_KEY"):
            chosen = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            chosen = "openai"
        elif os.getenv("GOOGLE_API_KEY"):
            chosen = "gemini"
        else:
            raise RuntimeError(
                "No teacher API key found. Set ONE of:\n"
                "    ANTHROPIC_API_KEY     (Claude Sonnet 4.5 — recommended)\n"
                "    OPENAI_API_KEY        (GPT-4.1 / GPT-4o)\n"
                "    GOOGLE_API_KEY        (Gemini 2.5 Pro)\n"
                "Or pass --provider explicitly."
            )

    if chosen == "anthropic":
        return TeacherClient(
            provider="anthropic",
            model=os.getenv("TEACHER_MODEL", "claude-sonnet-4-5"),
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.anthropic.com/v1/messages",
        )
    if chosen == "openai":
        return TeacherClient(
            provider="openai",
            model=os.getenv("TEACHER_MODEL", "gpt-4.1"),
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1/chat/completions",
        )
    if chosen == "gemini":
        return TeacherClient(
            provider="gemini",
            model=os.getenv("TEACHER_MODEL", "gemini-2.5-pro"),
            api_key=os.environ["GOOGLE_API_KEY"],
            base_url=(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{os.getenv('TEACHER_MODEL', 'gemini-2.5-pro')}:generateContent"
            ),
        )

    raise ValueError(f"unknown provider: {chosen}")
