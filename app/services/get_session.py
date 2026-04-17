from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

_memory: dict = {}


def get_memory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _memory:
        _memory[session_id] = InMemoryChatMessageHistory()
    return _memory[session_id]
