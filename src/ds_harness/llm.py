"""Minimal OpenAI-compatible Chat Completions client with bounded retries and JSON repair."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from ds_harness.config import LLMConfig, StrictModel, canonical_json
from ds_harness.errors import LLMError


class ChatClient:
    def __init__(self, config: LLMConfig, record: Callable[[dict], None], transport=None):
        self.config = config
        self.record = record
        self.calls = 0
        self.transport = transport

    def complete(self, messages: list[dict]) -> str:
        config = self.config
        base = os.environ.get(config.base_url_env, "").rstrip("/")
        model = os.environ.get(config.model_env, "")
        key = os.environ.get(config.api_key_env, "")
        if not base or not model:
            raise LLMError(f"Set {config.base_url_env} and {config.model_env}")
        parsed = urlparse(base)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise LLMError("Endpoint URL must not contain credentials, query strings or fragments")
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise LLMError("Use HTTPS for remote inference or HTTP for localhost")
        if len(canonical_json(messages)) > config.max_prompt_chars:
            raise LLMError("Prompt exceeded max_prompt_chars; reduce the task scope")
        headers = {}
        if key:
            headers[config.auth_header] = f"Bearer {key}" if config.auth_header == "Authorization" else key
        body = {
            "model": model, "messages": messages,
            config.token_parameter: config.max_output_tokens,
        }
        if config.json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(config.retries + 1):
            if self.calls >= config.max_calls:
                raise LLMError("LLM call budget exhausted")
            self.calls += 1
            # Count before sending so failed calls also consume the persisted budget.
            self.record({"event": "llm_attempt", "call": self.calls, "attempt": attempt + 1})
            try:
                with httpx.Client(timeout=config.timeout_seconds, transport=self.transport,
                                  follow_redirects=False) as client:
                    response = client.post(f"{base}/chat/completions", json=body, headers=headers)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < config.retries:
                        time.sleep(min(2**attempt, 8))
                        continue
                if not response.is_success:
                    raise LLMError(f"Inference request failed with HTTP {response.status_code}")
                payload = response.json()
                choice = payload["choices"][0]
                if choice.get("finish_reason") in {"length", "content_filter"}:
                    raise LLMError("Inference response was truncated or filtered")
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise LLMError("Inference response has no text content")
                usage = payload.get("usage") or {}
                self.record({"event": "llm_response", "call": self.calls, "usage": {
                    k: v for k, v in usage.items()
                    if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                    and isinstance(v, int)
                }})
                return content
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == config.retries:
                    raise LLMError("Inference endpoint timed out or could not be reached") from None
                time.sleep(min(2**attempt, 8))
            except (KeyError, IndexError, ValueError, TypeError):
                raise LLMError("Endpoint returned an invalid Chat Completions response") from None
        raise LLMError("Inference retries exhausted")

    def structured(self, system: str, context: dict, schema: type[StrictModel], validate=None):
        messages = [
            {"role": "system", "content": system + "\nReturn one JSON object matching this schema:\n"
             + canonical_json(schema.model_json_schema())},
            {"role": "user", "content": canonical_json(context)},
        ]
        for attempt in range(3):
            raw = self.complete(messages)
            try:
                result = schema.model_validate(json.loads(raw))
                if validate:
                    validate(result)
                return result
            except (ValidationError, ValueError, LLMError) as exc:
                if attempt == 2:
                    raise LLMError("LLM failed to produce a policy-valid plan after 3 attempts") from None
                # Do not persist raw output or echo possibly sensitive record values.
                if isinstance(exc, ValidationError):
                    errors = [{"location": list(e["loc"]), "type": e["type"]} for e in exc.errors()]
                else:
                    errors = [{"type": "policy_or_json", "reason": str(exc)[:500]}]
                messages.append({"role": "user", "content": "Regenerate the complete JSON. "
                                 + canonical_json({"validation_errors": errors})})
        raise LLMError("Structured inference failed")
