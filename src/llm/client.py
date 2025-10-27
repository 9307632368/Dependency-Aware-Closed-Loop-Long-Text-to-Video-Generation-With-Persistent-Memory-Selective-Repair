# src/llm/client.py
import os
import time
import json
import requests
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 2000
    timeout_s: int = 60
    retries: int = 2
    retry_sleep_s: float = 1.0


class LLMClient:
    """
    OpenAI-compatible Chat Completions client.
    Works with OpenAI and many OpenAI-compatible providers (local gateways too).
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")

    def chat(self, system: str, user: str) -> Tuple[bool, str, str]:
        """
        Returns: (ok, content, error_message)
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        last_err = ""
        for attempt in range(self.cfg.retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text}"
                    time.sleep(self.cfg.retry_sleep_s)
                    continue

                data = r.json()
                content = data["choices"][0]["message"]["content"]
                return True, content, ""
            except Exception as e:
                last_err = str(e)
                time.sleep(self.cfg.retry_sleep_s)

        return False, "", last_err


def load_llm_config_from_env(default_model: str = "gpt-4.1-mini") -> LLMConfig:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", default_model)

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or environment variables.")

    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        top_p=float(os.getenv("OPENAI_TOP_P", "0.9")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
        timeout_s=int(os.getenv("OPENAI_TIMEOUT_S", "60")),
        retries=int(os.getenv("OPENAI_RETRIES", "2")),
        retry_sleep_s=float(os.getenv("OPENAI_RETRY_SLEEP_S", "1.0")),
    )