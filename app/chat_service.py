from dataclasses import dataclass
from time import perf_counter

from app.llm.client import LLMClient, Message
from app.llm.errors import LLMClientError
from app.prompts.support_chat import SUPPORT_CHAT_PROMPT


CHAT_INSTRUCTION = SUPPORT_CHAT_PROMPT.render()


class ChatServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatReply:
    text: str
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str

class ChatService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def reply(self, user_text: str) -> ChatReply:
        text = user_text.strip()
        if not text:
            raise ChatServiceError('Сообщение не должно быть пустым.')

        messages = self._build_messages(text)

        started = perf_counter()

        try:
            llm_result = self._llm_client.generate(messages)
        except LLMClientError as error:
            raise ChatServiceError(
                f"Не удалось получить ответ модели: {error}"
            ) from error

        elapsed_seconds = perf_counter() - started

        if llm_result.finish_reason == "length":
            raise ChatServiceError("Ответ остановлен из-за ограничения длины.")
        if llm_result.finish_reason != "stop":
            raise ChatServiceError(
                "Модель не вернула готовый ответ. "
                f"Причина завершения: {llm_result.finish_reason}."
            )

        reply_text = self._validate_reply(llm_result.text)

        return ChatReply(
            text=reply_text,
            model=llm_result.model,
            prompt_id=SUPPORT_CHAT_PROMPT.prompt_id,
            prompt_version=SUPPORT_CHAT_PROMPT.version,
            elapsed_seconds=elapsed_seconds,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            total_tokens=llm_result.total_tokens,
            response_id=llm_result.response_id,
        )

    @staticmethod
    def _build_messages(user_text: str) -> list[Message]:
        return [
            {
                "role": "developer",
                "content": CHAT_INSTRUCTION,
            },
            {
                "role": "user",
                "content": (
                    "<customer_message>\n"
                    f"{user_text}\n"
                    "</customer_message>"
                ),
            }
        ]

    @staticmethod
    def _validate_reply(reply: str | None) -> str:
        if reply is None:
            raise ChatServiceError("Модель не вернула текст ответа.")
        cleaned_reply = reply.strip()
        if not cleaned_reply:
            raise ChatServiceError("Модель вернула пустой ответ.")

        return cleaned_reply