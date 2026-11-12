from app.prompts.support_chat.registry import (
    ACTIVE_SUPPORT_CHAT_VERSION,
    get_support_chat_prompt,
    list_support_chat_versions,
)

SUPPORT_CHAT_PROMPT = get_support_chat_prompt()

__all__ = [
    "ACTIVE_SUPPORT_CHAT_VERSION",
    "SUPPORT_CHAT_PROMPT",
    "get_support_chat_prompt",
    "list_support_chat_versions",
]
