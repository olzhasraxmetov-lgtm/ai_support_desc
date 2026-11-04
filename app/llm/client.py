from dataclasses import dataclass

from openai import OpenAI

from app.config import Settings


Message = dict[str, str]


@dataclass(frozen=True)
class LLMResult:
    text: str | None
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=str(settings.base_url),
            api_key=settings.llm_api_key.get_secret_value(),
        )

    def generate(self, messages: list[Message]) -> LLMResult:
        response = self._client.chat.completions.create(
            model=self._settings.model,
            messages=messages,
            temperature=self._settings.temperature,
            max_completion_tokens=self._settings.max_output_tokens,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResult(
            text=choice.message.content,
            model=response.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
            total_tokens=usage.total_tokens if usage is not None else None,
            response_id=response.id,
        )