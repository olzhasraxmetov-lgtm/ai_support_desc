from app.prompts.template import (
    PromptTemplate,
    PromptTemplateError,
)
from app.prompts.support_summary.v1_0_0 import (
    SUPPORT_SUMMARY_PROMPT_V1_0_0,
)
from app.prompts.support_summary.v1_1_0 import (
    SUPPORT_SUMMARY_PROMPT_V1_1_0,
)


ACTIVE_SUPPORT_SUMMARY_VERSION = "1.1.0"

_SUPPORT_SUMMARY_PROMPTS: dict[str, PromptTemplate] = {
    SUPPORT_SUMMARY_PROMPT_V1_0_0.version: SUPPORT_SUMMARY_PROMPT_V1_0_0,
    SUPPORT_SUMMARY_PROMPT_V1_1_0.version: SUPPORT_SUMMARY_PROMPT_V1_1_0,
}

def get_support_summary_prompt(
        version: str | None = None,
) -> PromptTemplate:
    selected_version = (
        ACTIVE_SUPPORT_SUMMARY_VERSION
        if version is None
        else version.strip()
    )

    try:
        return _SUPPORT_SUMMARY_PROMPTS[selected_version]
    except KeyError as error:
        available = ", ".join(_SUPPORT_SUMMARY_PROMPTS)
        raise PromptTemplateError(
            f"Неизвестная версия support_summary: "
            f"{selected_version}. "
            f"Доступны: {available}."
        ) from error

def list_support_summary_versions() -> tuple[str, ...]:
    return tuple(_SUPPORT_SUMMARY_PROMPTS)