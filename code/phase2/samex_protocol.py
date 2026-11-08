"""Keep task settings consistent across agent comparisons.

The protocol fixes the tools, task prompt, environment, turn limit, and token
budget. Models, execution methods, and connections between roles may vary.
SameXEnforcer checks the settings during a search."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class SameXProtocol:
    """Immutable fixed-condition bundle. frozen=True → cannot be mutated per-composition.

    TYPE-ADAPTIVE (author 2026-07-26): the fixed set adapts to benchmark type.
      - benchmark_type='agentic' (τ³/API-Bank/PlanBench/GAIA): fix env + tool_api +
        prompt + max_turns + token_budget; repository = N/A.
      - benchmark_type='coding' (reserved): fix repository + tools + prompt + max_turns +
        token_budget.
    harness is ALWAYS excluded from the fixed set (the evolved dimension = positioning A).
    """
    max_turns: int = 30                      # same interaction-step cap (all types)
    token_budget: int = 100_000              # same per-task token allowance (all types)
    prompt_template_id: str = "baseline_v1"  # same baseline prompt template (all types)
    benchmark_type: str = "agentic"          # 'agentic' | 'coding'
    # agentic-type fixed items:
    env_id: str = "benchmark_env_fixed"      # same environment (agentic)
    tool_api_id: str = "benchmark_tools_fixed"  # same tool-API set (agentic)
    # coding-type fixed items (reserved; N/A for agentic):
    repository_id: str = "N/A"
    tool_set_id: str = "N/A"
    # baseline prompt templates (same for all compositions; harness wraps its OWN reasoning
    # around these but the task framing is identical)
    system_prompt_interactive: str = (
        "You are an agent solving a task in an environment. Use the provided admissible "
        "actions/tools. Progress toward the goal.")
    system_prompt_static: str = (
        "Solve the task. Follow the required output format exactly.")

    def fixed_axes(self) -> dict:
        """Axes fixed for THIS benchmark type (audit basis). harness NOT included."""
        common = {"max_turns": self.max_turns, "token_budget": self.token_budget,
                  "prompt": self.prompt_template_id}
        if self.benchmark_type == "agentic":
            return {**common, "env": self.env_id, "tool_api": self.tool_api_id, "repository": "N/A"}
        if self.benchmark_type == "coding":
            return {**common, "repository": self.repository_id, "tools": self.tool_set_id, "env": "N/A"}
        return common

    def signature(self) -> str:
        ax = self.fixed_axes()
        return f"type={self.benchmark_type} " + " ".join(f"{k}={v}" for k, v in sorted(ax.items()))

    def as_dict(self):
        return asdict(self)


# The single canonical protocol instance for a search run. Every composition uses THIS.
DEFAULT_PROTOCOL = SameXProtocol()


class TokenBudgetExceeded(RuntimeError):
    pass


class SameXEnforcer:
    """Per-task enforcement + audit. Tracks token spend against the fixed budget and turn
    count against the fixed cap, and records the protocol signature seen by each
    composition so we can ASSERT it was constant across the whole search."""

    def __init__(self, protocol: SameXProtocol = DEFAULT_PROTOCOL):
        self.protocol = protocol
        self._seen_signatures = set()   # audit: must stay size 1 across a search
        self._audit_log = []

    def new_task_meter(self):
        """Return a fresh per-task meter (turns + tokens) bound to the fixed protocol."""
        return _TaskMeter(self.protocol)

    def audit(self, composition_id: str):
        """Record that this composition ran under the canonical protocol. Every
        composition must see the SAME signature → the set stays size 1."""
        sig = self.protocol.signature()
        self._seen_signatures.add(sig)
        self._audit_log.append({"composition": composition_id, "protocol_sig": sig})

    def assert_constant(self):
        """Raise if the fixed set was NOT constant across all audited compositions."""
        if len(self._seen_signatures) > 1:
            raise AssertionError(
                f"same-X violated: {len(self._seen_signatures)} distinct protocol "
                f"signatures across compositions: {self._seen_signatures}")
        return True

    def audit_summary(self):
        return {"protocol": self.protocol.as_dict(),
                "n_compositions_audited": len(self._audit_log),
                "distinct_signatures": list(self._seen_signatures),
                "constant": len(self._seen_signatures) <= 1}


class _TaskMeter:
    """Enforces max_turns + token_budget for ONE task, identically for every composition."""

    def __init__(self, protocol: SameXProtocol):
        self.max_turns = protocol.max_turns
        self.token_budget = protocol.token_budget
        self.turns = 0
        self.tokens = 0

    def can_step(self) -> bool:
        return self.turns < self.max_turns and self.tokens < self.token_budget

    def record_turn(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.turns += 1
        self.tokens += (prompt_tokens or 0) + (completion_tokens or 0)

    def budget_left(self) -> int:
        return max(0, self.token_budget - self.tokens)


if __name__ == "__main__":
    p = SameXProtocol()
    print("protocol signature:", p.signature())
    enf = SameXEnforcer(p)
    for comp in ["compA", "compB", "compC"]:
        enf.audit(comp)
        m = enf.new_task_meter()
        m.record_turn(100, 50)
        print(comp, "turns", m.turns, "tokens", m.tokens, "can_step", m.can_step())
    print("assert_constant:", enf.assert_constant())
    print("audit:", enf.audit_summary())
