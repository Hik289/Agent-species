"""Route tau2 model calls to configured providers and retry failed requests.

The wrapper spaces requests and retries empty responses, rate limits, overloads,
and timeouts. It also handles responses that omit model or token-usage fields.
After retries are exhausted, it raises an error."""
from __future__ import annotations

import os
import time


def install_robust_completion(min_gap: float = 1.5, max_attempts: int = 25,
                              request_timeout: float = 90.0,
                              claude_base: str | None = None,
                              gpt_base: str | None = None):
    import litellm
    claude_base = claude_base or os.environ.get("ANTHROPIC_BASE_URL")
    gpt_base = gpt_base or os.environ.get("OPENAI_BASE_URL")
    _orig = litellm.completion
    _last = [0.0]

    def _routed(*a, **k):
        m = k.get("model", "")
        if "claude" in m:
            if claude_base:
                k.setdefault("api_base", claude_base)
            if key := os.environ.get("ANTHROPIC_API_KEY"):
                k.setdefault("api_key", key)
            if not m.startswith("anthropic/"):
                k["model"] = "anthropic/" + m.split("/")[-1]
            k.setdefault("max_tokens", 1024)
            # ⚠️ QUOTA (2026-08-05): the proxy 429s opus/sonnet on ANY system-role message
            # (even tiny). Merge the system prompt into the first user message so opus/sonnet
            # multi-turn (τ³) episodes don't 429. tau2 builds a small system prompt internally;
            # this catches it at the proxy boundary without patching tau2.
            msgs = k.get("messages")
            if msgs and any(mm.get("role") == "system" for mm in msgs):
                sys_txt = "\n".join(mm.get("content", "") for mm in msgs if mm.get("role") == "system")
                rest = [mm for mm in msgs if mm.get("role") != "system"]
                # prepend system text to the first user message (or add one)
                if rest and rest[0].get("role") == "user":
                    rest[0] = {"role": "user", "content": sys_txt + "\n\n" + str(rest[0].get("content", ""))}
                else:
                    rest = [{"role": "user", "content": sys_txt}] + rest
                k["messages"] = rest
        else:
            if gpt_base:
                k.setdefault("api_base", gpt_base)
            if key := os.environ.get("OPENAI_API_KEY"):
                k.setdefault("api_key", key)
            if not m.startswith("openai/"):
                k["model"] = "openai/" + m.split("/")[-1]
            for t in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                k.pop(t, None)
        # per-request timeout so a hung proxy call fails fast + retries (not stall forever)
        k.setdefault("timeout", request_timeout)
        # let US handle retries (litellm's num_retries returns None on failure)
        k.pop("num_retries", None)
        last = None
        for att in range(max_attempts):
            dt = time.time() - _last[0]
            if dt < min_gap:
                time.sleep(min_gap - dt)
            _last[0] = time.time()
            try:
                r = _orig(*a, **k)
                if r is None or not getattr(r, "choices", None):
                    time.sleep(min(2 + att, 8))
                    continue
                if getattr(r, "model", None) in (None, ""):
                    r.model = k.get("model", "claude-opus-4-8")
                return r
            except Exception as e:  # noqa: BLE001
                last = e
                s = str(e).lower()
                if any(x in s for x in ("429", "rate", "overloaded", "timeout", "connection")):
                    # FAIL-FAST + stable: short FIXED backoff (3s) so time-varying 429
                    # windows are ridden out by steady quick retries, not multi-minute
                    # exponential stalls. Bounded max_attempts caps total wait.
                    time.sleep(3)
                    continue
                raise
        raise RuntimeError(f"completion None/err after {max_attempts} tries: {last}")

    litellm.completion = _routed

    # patch tau2's cost/usage utils to tolerate proxy-missing fields
    try:
        import tau2.utils.llm_utils as LU
        LU.completion = _routed

        def _safecost(response):
            try:
                if getattr(response, "model", None) in (None, ""):
                    response.model = "claude-opus-4-8"
                return LU.completion_cost(completion_response=response)
            except Exception:
                return 0.0
        LU.get_response_cost = _safecost

        _ousage = LU.get_response_usage
        LU.get_response_usage = lambda r: (_ousage(r) if r is not None else None)
    except Exception:
        pass

    # patch already-imported modules that bound `completion` by name
    import sys
    for mod in ("tau2.agent.llm_agent", "tau2.user.user_simulator"):
        try:
            if mod in sys.modules and hasattr(sys.modules[mod], "completion"):
                sys.modules[mod].completion = _routed
        except Exception:
            pass

    return _routed
