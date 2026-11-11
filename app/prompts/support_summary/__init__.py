from app.prompts.support_summary.registry import (
    ACTIVE_SUPPORT_SUMMARY_VERSION,
    get_support_summary_prompt,
    list_support_summary_versions,
)


SUPPORT_SUMMARY_PROMPT = get_support_summary_prompt()


__all__ = [
    "ACTIVE_SUPPORT_SUMMARY_VERSION",
    "SUPPORT_SUMMARY_PROMPT",
    "get_support_summary_prompt",
    "list_support_summary_versions",
]