"""API-Bank benchmark wrapper for Phase 2 (agentic tool-use, non-docker, local mock).

Level-3 stateful multi-turn tool dialogue. Each "task" = one API decision point: given
the conversation history + available API descriptions, the agent (via harness) must
output the correct next API call. Scored deterministically by parsing the call and
comparing API name + params to ground truth (via API-Bank's evaluator + parse_api_call).

Same-X: tool set (73 mock APIs) + prompt template + max_turns + token_budget FIXED
(agentic type); harness varies how the call is produced.

Reuses the API-Bank repo (configured via APIBANK_DATA) evaluator machinery.
"""
from __future__ import annotations

import os
import sys
import glob

_APIBANK_DIR = os.path.abspath(os.path.expanduser(os.environ.get("APIBANK_DATA") or "./benchmarks/api-bank"))


def _ensure_apibank():
    if _APIBANK_DIR not in sys.path:
        sys.path.insert(0, _APIBANK_DIR)
    # evaluator + parse expect cwd at the repo (loads ./apis, ./lv3-samples)
    if os.getcwd() != _APIBANK_DIR:
        os.chdir(_APIBANK_DIR)


_LV3_GLOB = os.path.join(_APIBANK_DIR, "lv1-lv2-samples", "level-2-toolsearcher", "*level-3*.jsonl")


# each task ref = (dialogue_file, sample_index)
def list_problems(n=0, seed=0, max_samples_per_dialogue=3):
    _ensure_apibank()
    import json
    from evaluator import Sample
    dialogues = sorted(glob.glob(_LV3_GLOB))
    refs = []
    for df in dialogues:
        try:
            chat_history = _parse_dialogue(df)
            samples = Sample.from_chat_history(chat_history)[:max_samples_per_dialogue]
            for si in range(len(samples)):
                refs.append((df, si))
        except Exception:
            continue
    if seed:
        import random
        random.Random(seed).shuffle(refs)
    if n:
        refs = refs[:n]
    return [("apibank", r, 30) for r in refs]


def _parse_dialogue(path):
    """Load a level-3 JSONL dialogue into API-Bank chat_history format (list of dicts)."""
    import json
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class APIBankTask:
    kind = "static"

    def __init__(self, ref, max_steps: int = 1):
        _ensure_apibank()
        from evaluator import Sample, Evaluator
        self.dialogue_file, self.sample_idx = ref
        chat_history = _parse_dialogue(self.dialogue_file)
        self.samples = Sample.from_chat_history(chat_history)
        self.evaluator = Evaluator(self.samples)
        self.sample_id = self.sample_idx

    def prompt(self) -> str:
        api_descriptions, chat_history = self.evaluator.get_model_input(self.sample_id)
        history_str = ""
        for turn in chat_history:
            role = turn.get("role")
            if role == "User":
                history_str += f"User: {turn.get('text','')}\n"
            elif role == "AI":
                history_str += f"AI: {turn.get('text','')}\n"
            elif role == "API":
                history_str += (f"API call: {turn.get('api_name','?')}({turn.get('param_dict',{})})\n"
                                f"API result: {turn.get('result','?')}\n")
        return (f"You are an API-calling agent.\n\nAvailable APIs:\n{api_descriptions}\n\n"
                f"Conversation so far:\n{history_str}\n"
                "Output the next step. If an API call is needed, use the format "
                "ApiName(param1=value1, param2=value2). Otherwise respond to the user.")

    def score(self, answer: str) -> int:
        _ensure_apibank()
        try:
            from api_call_extraction import parse_api_call
            gt = self.evaluator.dataset[self.sample_id].ground_truth
            gt_role = gt.get("role")
            if gt_role == "API":
                try:
                    parsed = parse_api_call(answer)
                except Exception:
                    return 0
                if parsed and parsed.get("api_name") == gt.get("api_name"):
                    gt_params = gt.get("param_dict", {})
                    pred_params = parsed.get("param_dict", {})
                    # correct API + matching param keys = pass
                    if set(pred_params.keys()) == set(gt_params.keys()):
                        return 1
                    return 0
                return 0
            else:
                # AI response: ROUGE-L threshold
                try:
                    from rouge import Rouge
                    ref = gt.get("text", "")
                    if ref and answer:
                        r = Rouge().get_scores(answer[:500], ref[:500])[0]["rouge-l"]["f"]
                        return 1 if r > 0.5 else 0
                except Exception:
                    return 0
                return 0
        except Exception:
            return 0

    def reset(self):
        return {"observation": self.prompt(), "goal": "", "admissible_commands": [],
                "inventory": ""}, ""

    def close(self):
        pass


if __name__ == "__main__":
    probs = list_problems(n=3)
    print("apibank tasks:", len(probs), "sample:", probs[0] if probs else None)
    if probs:
        t = APIBankTask(probs[0][1])
        print("prompt head:", t.prompt()[:250])
        gt = t.evaluator.dataset[t.sample_id].ground_truth
        print("gt role:", gt.get("role"), "api:", gt.get("api_name"))
