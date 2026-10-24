"""Conversation history with trimming."""

from .config import LLMConfig


class ConversationManager:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._history: list[dict[str, str]] = []

    def add_user(self, text: str) -> None:
        # A pause mid-thought splits speech into two utterances ("It's great."
        # ... "How about you?"); if the first got no reply, keep them as one
        # message so the model answers the whole thing.
        if self._history and self._history[-1]["role"] == "user":
            self._history[-1]["content"] += " " + text
        else:
            self._history.append({"role": "user", "content": text})

    def add_assistant(self, text: str, interrupted: bool = False) -> None:
        # Store only what was actually spoken; no meta-marker, or the model
        # starts narrating the interruption ("you broke the connection").
        if text.strip():
            self._history.append({"role": "assistant", "content": text.strip()})

    def last_assistant(self) -> str | None:
        for msg in reversed(self._history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    def unreplied_count(self) -> int:
        """Assistant messages since the user last spoke."""
        n = 0
        for msg in reversed(self._history):
            if msg["role"] != "assistant":
                break
            n += 1
        return n

    def messages(self, resume_hint: str | None = None,
                 memory_block: str | None = None,
                 tools_block: str | None = None,
                 now_block: str | None = None) -> list[dict[str, str]]:
        max_msgs = self.cfg.history_max_turns * 2
        trimmed = self._history[-max_msgs:]
        msgs = [{"role": "system", "content": self.cfg.system_prompt}]
        if tools_block:  # persistent tool catalogue, right after the persona
            msgs.append({"role": "system", "content": tools_block})
        # Current date/time: after the unchanging persona/tools so local servers
        # can still reuse their cached prompt prefix.
        if now_block:
            msgs.append({"role": "system", "content": now_block})
        if memory_block:  # recalled long-term context, before the recent turns
            msgs.append({"role": "system", "content": memory_block})
        msgs += trimmed
        if resume_hint:  # transient nudge for this turn only, not persisted
            msgs.append({"role": "system", "content": resume_hint})
        return msgs

    def clear(self) -> None:
        self._history.clear()
