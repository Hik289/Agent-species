"""Run a tau2 agent using the benchmark's native tool-calling interface.

The agent supports direct answers, chain-of-thought reasoning, ReAct, and
Reflexion. It keeps the task policy, tools, and step limit fixed across model
choices. The bridge installs API retry handling before tasks run."""
from __future__ import annotations

from typing import List, Optional

from tau2.agent.llm_agent import LLMAgent, LLMAgentState, AGENT_INSTRUCTION
from tau2.environment.tool import Tool


# --- Harness reasoning-scaffolds (the evolved dimension) -----------------------------
# Each scaffold is appended to the base agent instruction. They are written to induce
# GENUINELY different agent behavior under native function-calling (not collapse to
# identical tool calls). Verified empirically before use in the search.
HARNESS_SCAFFOLD = {
    "direct": (
        "\n\n<harness=direct>\n"
        "Operate in DIRECT mode. Take the single best next action immediately. Do NOT "
        "narrate your reasoning or restate the situation. When a tool is needed, call it "
        "right away with no preamble. Keep any user-facing message to one short sentence.\n"
        "</harness>"
    ),
    "cot": (
        "\n\n<harness=chain_of_thought>\n"
        "Operate in CHAIN-OF-THOUGHT mode. Before each tool call, think through the "
        "problem step by step: (1) what the user needs, (2) what information/authentication "
        "is still missing, (3) which single tool advances the goal, (4) the exact arguments. "
        "Work through these steps explicitly in your reasoning, THEN make the tool call.\n"
        "</harness>"
    ),
    "react": (
        "\n\n<harness=react>\n"
        "Operate in ReAct mode (Reason+Act). At every step, first form an explicit Thought "
        "about the current state — what is known, what the last observation (tool result or "
        "user message) told you, and what remains — then take exactly one Action (a tool call "
        "or a user message). Treat each tool result as an Observation that updates your next "
        "Thought. Interleave Thought and Action every turn.\n"
        "</harness>"
    ),
    "reflexion": (
        "\n\n<harness=reflexion>\n"
        "Operate in REFLEXION mode. After every tool result, briefly self-critique: did the "
        "result match what you expected? If a tool call FAILED, returned an error, or gave an "
        "unexpected result, explicitly diagnose WHY (wrong arguments? missing prerequisite? "
        "wrong tool?) and correct your approach before retrying. Do not repeat a failing call "
        "unchanged. Verify each step succeeded before moving on.\n"
        "</harness>"
    ),
}

# Small per-harness llm_args nudges (kept conservative; the scaffold is the main lever).
HARNESS_LLM_ARGS = {
    "direct": {"temperature": 0.0},
    "cot": {"temperature": 0.3},
    "react": {"temperature": 0.3},
    "reflexion": {"temperature": 0.4},
}


# SAME-X FIXED COMPONENT (orchestrator guardrail): policy-delivery placement is a fixed,
# uniform prompt-axis component applied IDENTICALLY to every composition (all models,
# all harnesses). "user" = domain policy delivered as the first user message (workaround
# for the proxy's large-system-prompt 429 quota); "system" = policy in system prompt
# (original). MUST be constant across the whole search — never varies by genotype.
POLICY_PLACEMENT = "user"  # uniform for all comps; part of the fixed same-X "prompt" axis

_POLICY_USER_PREFIX = (
    "Here is the policy you must follow for this conversation. Follow it strictly as if "
    "it were your system instructions:\n\n<policy>\n{policy}\n</policy>\n\n"
    "Acknowledge silently and begin assisting the user according to this policy."
)


class NativeHarnessAgent(LLMAgent):
    """tau2 native-function-calling LLMAgent with a harness reasoning-scaffold gene.

    Policy placement (system vs user) is a FIXED same-X component (POLICY_PLACEMENT),
    uniform across all comps — not a genotype axis.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        harness: str = "react",
        llm_args: Optional[dict] = None,
        policy_placement: str = POLICY_PLACEMENT,
    ):
        self.harness = harness if harness in HARNESS_SCAFFOLD else "react"
        self.policy_placement = policy_placement
        # merge harness llm_args nudge with any caller overrides (caller wins)
        merged = dict(HARNESS_LLM_ARGS.get(self.harness, {}))
        if llm_args:
            merged.update(llm_args)
        # ⚠️ The newer reasoning models (opus-4-8, opus-5, sonnet-5) REJECT the
        # `temperature` param ("temperature is deprecated for this model"). Strip it for
        # those tiers so the harness runs on all models (the scaffold is the main harness
        # lever anyway). Only sonnet-4-6 / haiku-4-5 accept temperature.
        if any(t in llm for t in ("opus-4-8", "opus-5", "sonnet-5")):
            merged.pop("temperature", None)
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=merged,
        )

    @property
    def agent_instruction(self) -> str:
        return AGENT_INSTRUCTION + HARNESS_SCAFFOLD.get(self.harness, "")

    @property
    def system_prompt(self) -> str:
        from tau2.agent.llm_agent import SYSTEM_PROMPT
        if self.policy_placement == "user":
            # SHORT system prompt (instruction + harness scaffold only); policy goes to
            # the first user message via get_init_state → avoids large-system 429.
            return (
                "<instructions>\n" + self.agent_instruction + "\n</instructions>\n"
                "The domain policy will be provided in the first user message; follow it "
                "strictly for the entire conversation."
            )
        # original: policy in system prompt
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=self.agent_instruction,
        )

    def get_init_state(self, message_history=None):
        state = super().get_init_state(message_history=message_history)
        if self.policy_placement == "user":
            # inject the domain policy as the FIRST user message (before any dialogue).
            from tau2.data_model.message import UserMessage
            policy_msg = UserMessage(
                role="user",
                content=_POLICY_USER_PREFIX.format(policy=self.domain_policy),
            )
            state.messages = [policy_msg] + list(state.messages)
        return state
