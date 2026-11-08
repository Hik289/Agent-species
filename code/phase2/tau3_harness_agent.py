"""Run a tau2 agent that requests tools through text commands.

The agent reads the task policy and available tools. It parses the last
ACTION directive as a tool call or the last RESPOND directive as a user reply.
Execution methods control how the model reasons before choosing a command."""
from __future__ import annotations

import json
import re

# routing patch applied by the bridge before import; here just use litellm.completion
import litellm

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import (
    AssistantMessage, ToolCall, SystemMessage, UserMessage, ToolMessage, MultiToolMessage,
)


HARNESS_STYLE = {
    "direct": "Decide the single next action. Output ONLY one directive.",
    "cot": "Think step by step briefly, then on the LAST line output ONE directive.",
    "react": "Reason about the state (Thought), then output ONE directive (Action).",
    "reflexion": "Reason; if the last tool result indicates failure, reflect and correct; output ONE directive.",
}

_PROTOCOL = (
    "Respond with EXACTLY ONE directive on its own line:\n"
    "  ACTION: <tool_name>({<json arguments>})   -- to call a tool\n"
    "  RESPOND: <message>                          -- to talk to the user\n"
    "Use a tool from the available list. Arguments must be valid JSON."
)


def _tools_text(tools):
    lines = []
    for t in tools:
        schema = getattr(t, "openai_schema", None)
        if schema and "function" in schema:
            fn = schema["function"]
            params = fn.get("parameters", {}).get("properties", {})
            pnames = ", ".join(params.keys())
            lines.append(f"- {fn['name']}({pnames}): {fn.get('description','')[:120]}")
        else:
            lines.append(f"- {getattr(t, 'name', str(t))}")
    return "\n".join(lines)


def _parse_directive(text):
    """Parse the last ACTION:/RESPOND: directive. Returns ('action', name, args) or ('respond', msg) or None."""
    if not text:
        return None
    # find all directives, take the last
    actions = list(re.finditer(r"ACTION:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$",
                               text, re.MULTILINE | re.DOTALL))
    responds = list(re.finditer(r"RESPOND:\s*(.+?)$", text, re.MULTILINE | re.DOTALL))
    last_action = actions[-1] if actions else None
    last_respond = responds[-1] if responds else None
    # choose whichever appears later in the text
    if last_action and (not last_respond or last_action.start() > last_respond.start()):
        name = last_action.group(1)
        arg_str = last_action.group(2).strip()
        try:
            args = json.loads(arg_str) if arg_str else {}
            if not isinstance(args, dict):
                args = {}
        except Exception:
            # tolerant parse: key=value pairs
            args = {}
            for m in re.finditer(r"(\w+)\s*=\s*\"?([^,\"]+)\"?", arg_str):
                args[m.group(1)] = m.group(2).strip()
        return ("action", name, args)
    if last_respond:
        return ("respond", last_respond.group(1).strip())
    return None


class HarnessAgent(HalfDuplexAgent):
    """LLM agent with a text-based tool harness (no native function-calling)."""

    def __init__(self, tools, domain_policy, llm, harness="react", llm_args=None):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.harness = harness
        self.llm_args = llm_args or {}
        self._sys = self._build_system()

    def _build_system(self):
        style = HARNESS_STYLE.get(self.harness, HARNESS_STYLE["react"])
        return (f"You are a customer-service agent. Follow this policy strictly:\n"
                f"{self.domain_policy}\n\nAvailable tools:\n{_tools_text(self.tools)}\n\n"
                f"{style}\n{_PROTOCOL}")

    def get_init_state(self, message_history=None):
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self._sys)],
            messages=message_history or [],
        )

    def generate_next_message(self, message, state):
        # append incoming message(s) to state
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)
        # build litellm messages (text only)
        lm = [{"role": "system", "content": self._sys}]
        for m in state.messages:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
            if isinstance(m, ToolMessage):
                lm.append({"role": "user", "content": f"[TOOL RESULT] {content}"})
            elif role in ("assistant",):
                # include prior tool calls as text
                tc = getattr(m, "tool_calls", None)
                if tc:
                    content += " " + " ".join(f"ACTION: {c.name}({json.dumps(c.arguments)})" for c in tc)
                lm.append({"role": "assistant", "content": content})
            else:
                lm.append({"role": "user", "content": content})
        # ⚠️ Anthropic requires the first non-system message to be role=user (tau2's agent
        # greets first → conversation starts with assistant → Anthropic 400/empty → this
        # was the τ³=0 root cause). Fix: drop leading assistant turns; ensure user-first.
        convo = lm[1:]  # everything after system
        while convo and convo[0]["role"] == "assistant":
            convo.pop(0)
        if not convo or convo[0]["role"] != "user":
            convo = [{"role": "user", "content": "(start)"}] + convo
        # also merge consecutive same-role messages (Anthropic disallows) → alternate
        merged = []
        for msg in convo:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n" + msg["content"]
            else:
                merged.append(dict(msg))
        lm = [lm[0]] + merged
        # call LLM (routed to proxy by the bridge's litellm patch). Claude needs max_tokens.
        args = dict(self.llm_args)
        args.setdefault("max_tokens", 1024)
        # RETRY 429/rate-limit (do NOT swallow → empty text → fallback → reward 0).
        import time as _t
        text = ""
        last = None
        for attempt in range(6):
            try:
                resp = litellm.completion(model=self.llm, messages=lm, **args)
                if resp is not None and getattr(resp, "choices", None):
                    msg0 = resp.choices[0].message
                    text = (getattr(msg0, "content", None) or "")
                break
            except Exception as e:  # noqa: BLE001
                last = e
                if "429" in str(e) or "rate" in str(e).lower():
                    _t.sleep(min(2 ** (attempt + 1), 40))
                    continue
                break  # non-429 error → give up (text stays "")
        directive = _parse_directive(text)
        if directive and directive[0] == "action":
            _, name, args = directive
            am = AssistantMessage(role="assistant", content=None,
                                  tool_calls=[ToolCall(id="tc", name=name, arguments=args)])
        else:
            msg = directive[1] if directive else text.strip()[:500]
            am = AssistantMessage(role="assistant", content=msg or "How can I help?")
        state.messages.append(am)
        return am, state
