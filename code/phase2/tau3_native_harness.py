"""τ³-bench (tau2) agent with NATIVE function-calling + evolved HARNESS dimension.

Replaces the earlier TEXT-tool HarnessAgent. COMPARE (2026-08-01) proved the Claude
compatible provider endpoint fully supports native Anthropic function-calling, so the text-parse
workaround (a GPT-era necessity) is unnecessary and less valid. Native function-calling
is what τ³ was DESIGNED to measure.

Design (orchestrator-approved):
  * Subclass tau2's native LLMAgent → keeps native tool-calling via generate().
  * The HARNESS gene (direct / cot / react / reflexion) is injected as a reasoning-
    SCAFFOLD in the system prompt (+ small llm_args differences), so harness still
    varies as an evolved dimension (positioning-A thesis intact) — but now over native
    tools instead of text-parse.
  * Same-X: tools + domain policy + max_steps FIXED by tau2; HARNESS is what varies.

⚠️ The 4 harnesses MUST produce measurably DIFFERENT agent behavior on τ³ (verified via
   scripts/verify_harness_differs.py) — otherwise "harness co-evolution" has nothing to
   evolve on. Scaffolds are written to induce real behavioral differences:
     - direct   : immediate action, no deliberation, terse
     - cot      : explicit step-by-step reasoning in message content before acting
     - react    : Thought→Action interleaving, verbalize state each step
     - reflexion: self-critique after each tool result; on failure, diagnose then retry

The robust None/429 retry wrapper is installed by the bridge (install_robust_completion)
so tau2's generate() never receives None from the proxy.
"""
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
