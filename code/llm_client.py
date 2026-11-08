"""Call model APIs with retries, token tracking, and spending limits.

LLMClient records usage by model and purpose. BudgetExceeded stops a run when
recorded spending exceeds its configured limit. API access is configured through
local environment variables."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

MODEL = "gpt-5.4-mini"  # default GPT model

def _is_claude(model: str) -> bool:
    return bool(model) and "claude" in model.lower()

# --- Pricing (USD per 1M token, OpenAI-compatible published rates for gpt-5.4-mini class) ---
#
PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60

# --- Hard caps  ---
DEFAULT_BUDGET_HARD_CAP = 500.0  # USD
DEFAULT_BUDGET_WARN_THRESHOLD = 400.0  # USD


class BudgetExceeded(RuntimeError):
    """Raised when accumulated USD spend exceeds hard cap. ."""


@dataclass
class UsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_retries: int = 0
    total_usd: float = 0.0
    by_purpose: dict[str, dict[str, float]] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def add(self, prompt: int, completion: int, purpose: str, model: str | None = None) -> float:
        # model-aware pricing (per 1M tokens in/out). Adjust if proxy bills differ.
        m = (model or "").lower()
        pin, pout = PRICE_INPUT_PER_1M, PRICE_OUTPUT_PER_1M  # gpt-5.4-mini default 0.15/0.60
        if model == "gpt-5.4":
            pin, pout = 1.25, 5.0
        elif model == "gpt-5.5":
            pin, pout = 2.5, 10.0
        # Claude tiers (author Claude-only 2026-07-31). opus-5/sonnet-5 = new flagships.
        elif "opus-5" in m:                          # claude-opus-5 (top flagship)
            pin, pout = 15.0, 75.0
        elif "sonnet-5" in m:                        # claude-sonnet-5 (new mid-high)
            pin, pout = 3.0, 15.0
        elif "opus" in m:                            # claude-opus-4-8
            pin, pout = 15.0, 75.0
        elif "sonnet" in m:                          # claude-sonnet-4-6
            pin, pout = 3.0, 15.0
        elif "haiku" in m:                           # claude-haiku-4-5 (cheap)
            pin, pout = 0.80, 4.0
        cost = (prompt / 1_000_000) * pin + (completion / 1_000_000) * pout
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_calls += 1
        self.total_usd += cost
        bucket = self.by_purpose.setdefault(purpose, {"prompt": 0, "completion": 0, "calls": 0, "usd": 0.0})
        bucket["prompt"] += prompt
        bucket["completion"] += completion
        bucket["calls"] += 1
        bucket["usd"] += cost
        self.last_updated = time.time()
        return cost


class LLMClient:
    """Thread-safe OpenAI-compatible gpt-5.4-mini client with retry + tracker.

    Args:
        budget_usd_hard_cap: 累计 USD 上限 (default 500). 超过 raise BudgetExceeded.
        tracker_path: JSON file 持久化 usage; 每次 chat 后增量写.
        purpose_tag: 默认 purpose tag (可在 chat call 覆盖).
        max_retries: rate-limit + transient 失败 retry 次数.
    """

    def __init__(
        self,
        budget_usd_hard_cap: float = DEFAULT_BUDGET_HARD_CAP,
        tracker_path: str | os.PathLike = "experiments/budget_tracker.json",
        purpose_tag: str = "default",
        max_retries: int = 5,
    ) -> None:
        self._client = None  # Initialize only the provider used by a call.
        self._anthropic = None  # lazy-init on first Claude call
        self._budget_cap = float(budget_usd_hard_cap)
        self._tracker_path = Path(tracker_path)
        self._tracker_path.parent.mkdir(parents=True, exist_ok=True)
        self._purpose_tag = purpose_tag
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._usage = self._load_tracker()

    # ---- tracker IO ----
    def _load_tracker(self) -> UsageRecord:
        if self._tracker_path.exists():
            try:
                raw = json.loads(self._tracker_path.read_text())
                return UsageRecord(**raw)
            except Exception:  # noqa: BLE001
                pass
        return UsageRecord()

    def _persist_tracker(self) -> None:
        tmp = self._tracker_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self._usage), indent=2))
        tmp.replace(self._tracker_path)

    # ---- accessors ----
    @property
    def total_usd(self) -> float:
        return self._usage.total_usd

    def summary(self) -> dict[str, Any]:
        return asdict(self._usage)

    def check_budget(self) -> None:
        if self._usage.total_usd >= self._budget_cap:
            raise BudgetExceeded(
                f"Hard cap ${self._budget_cap:.2f} exceeded; spent ${self._usage.total_usd:.2f}"
            )

    # ---- main API ----
    def chat_model(self, model: str, messages, max_tokens: int = 1024,
                   temperature: float = 0.0, purpose: str | None = None, **extra):
        """chat() with an explicit model override (Phase 0 multi-model compositions)."""
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature,
                         purpose=purpose, _model=model, **extra)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        purpose: str | None = None,
        _model: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """One chat completion with retry + tracking.

        Returns dict {text, prompt_tokens, completion_tokens, cost_usd, raw}.
        Raises BudgetExceeded on over-budget; APIConnectionError / RateLimitError after max_retries.
        """
        self.check_budget()
        purpose = purpose or self._purpose_tag
        use_model = _model or MODEL

        if _is_claude(use_model):
            return self._chat_anthropic(messages, use_model, max_tokens, temperature, purpose)

        if self._client is None:
            self._client = OpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL") or None,
                api_key=os.environ.get("OPENAI_API_KEY") or None,
            )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                # The archived OpenAI-compatible interface relies on the server's token limit.
                call_kwargs = dict(extra)
                _ = max_tokens  # intentionally unused with local proxy
                # gpt-5.4-mini only supports default temperature; do not pass unless explicitly non-default
                if temperature != 0.0:
                    call_kwargs["temperature"] = temperature
                resp = self._client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    **call_kwargs,
                )
                text = resp.choices[0].message.content if resp.choices else ""
                usage = resp.usage
                p_tok = getattr(usage, "prompt_tokens", 0) or 0
                c_tok = getattr(usage, "completion_tokens", 0) or 0
                with self._lock:
                    cost = self._usage.add(p_tok, c_tok, purpose, model=use_model)
                    self._persist_tracker()
                return {
                    "text": text,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "model": use_model,
                    "cost_usd": cost,
                    "attempt": attempt,
                }
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_exc = e
                with self._lock:
                    self._usage.total_retries += 1
                # exponential backoff: 2, 4, 8, 16, 32 s, jitter ±25%
                base = 2 ** (attempt + 1)
                sleep_s = base * (0.75 + 0.5 * (hash((time.time(), attempt)) & 0xFFFF) / 0xFFFF)
                time.sleep(min(sleep_s, 60.0))
            except APIStatusError as e:
                # 4xx 不 retry (config error); 5xx retry
                if 500 <= e.status_code < 600 and attempt < self._max_retries:
                    last_exc = e
                    time.sleep(2 ** (attempt + 1))
                    continue
                with self._lock:
                    self._usage.total_failures += 1
                    self._persist_tracker()
                raise
            except Exception:  # noqa: BLE001
                with self._lock:
                    self._usage.total_failures += 1
                    self._persist_tracker()
                raise

        with self._lock:
            self._usage.total_failures += 1
            self._persist_tracker()
        assert last_exc is not None
        raise last_exc

    def _chat_anthropic(self, messages, model, max_tokens, temperature, purpose):
        """Call the Messages SDK with the system message separated."""
        if self._anthropic is None:
            from anthropic import Anthropic
            self._anthropic = Anthropic(
                base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
                api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
                max_retries=self._max_retries,
            )
        # split out system message(s); anthropic takes system as a top-level param
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        conv = [{"role": ("assistant" if m["role"] == "assistant" else "user"),
                 "content": m["content"]} for m in messages if m.get("role") != "system"]
        if not conv:
            conv = [{"role": "user", "content": " "}]
        kwargs = dict(model=model, max_tokens=max_tokens, messages=conv)
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if temperature != 0.0:
            kwargs["temperature"] = temperature
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._anthropic.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text") \
                    if resp.content else ""
                p_tok = getattr(resp.usage, "input_tokens", 0) or 0
                c_tok = getattr(resp.usage, "output_tokens", 0) or 0
                with self._lock:
                    cost = self._usage.add(p_tok, c_tok, purpose, model=model)
                    self._persist_tracker()
                return {"text": text, "prompt_tokens": p_tok, "completion_tokens": c_tok,
                        "cost_usd": cost, "model": model, "attempt": attempt}
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt < self._max_retries:
                    time.sleep(min(2 ** (attempt + 1), 60.0))
                    continue
                with self._lock:
                    self._usage.total_failures += 1
                    self._persist_tracker()
                raise
        assert last_exc is not None
        raise last_exc


if __name__ == "__main__":
    # M1 smoke test (per the design spec B3)
    import sys

    client = LLMClient(
        budget_usd_hard_cap=500.0,
        tracker_path=os.environ.get("LLM_TRACKER", "./results/llm_smoke.json"),
        purpose_tag="m1_smoke",
    )

    print("=== OpenAI-compatible gpt-5.4-mini smoke test (5 calls) ===")
    t0 = time.time()
    latencies = []
    for i in range(5):
        t_call = time.time()
        try:
            out = client.chat(
                messages=[
                    {"role": "system", "content": "You are a terse assistant."},
                    {"role": "user", "content": f"Say only OK#{i}. No other text."},
                ],
                max_tokens=20,
            )
            dt = time.time() - t_call
            latencies.append(dt)
            print(f"  call#{i}: {dt:.2f}s, p={out['prompt_tokens']} c={out['completion_tokens']} "
                  f"cost=${out['cost_usd']:.6f} text={out['text']!r}")
        except Exception as e:
            print(f"  call#{i}: FAIL {type(e).__name__}: {e}")
            sys.exit(1)

    avg = sum(latencies) / len(latencies)
    print(f"\nTotal: {time.time()-t0:.1f}s, avg latency {avg:.2f}s, "
          f"spent ${client.total_usd:.6f}")
    print("PASS" if avg < 8.0 else "WARN: avg latency > 8s")
