from dataclasses import dataclass, field
from typing import Any
from app.auth import User

@dataclass
class ChatSession:
    conversation_id: str
    user: User | None = None
    thread: Any = None
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)

    def start_turn(self) -> None:
        self.citations = []
        self.tool_calls = []

_SESSIONS: dict[str, ChatSession] = {}

def get_session(conversation_id: str) -> ChatSession:
    if conversation_id not in _SESSIONS:
        _SESSIONS[conversation_id] = ChatSession(conversation_id=conversation_id)
    return _SESSIONS[conversation_id]
