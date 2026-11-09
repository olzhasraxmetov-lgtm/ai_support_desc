from dataclasses import dataclass
from time import perf_counter

from app.llm.client import LLMClient, Message
from app.llm.errors import LLMClientError


DEVELOPER_INSTRUCTION = """
Ты помогаешь оператору службы поддержки кратко пересказывать
обращения клиентов.

Цель:
Сформулируй основную проблему клиента одним предложением.

Правила:
- Используй только факты, явно указанные в обращении.
- Если они указаны, сохраняй номер заказа, дату и название товара.
- Не добавляй советы, решения, обещания и оценки.
- Если деталь отсутствует или неясна, опусти ее.

Порядок работы:
1. Определи основную проблему клиента.
2. Выбери только факты, необходимые для понимания проблемы.
3. Удали приветствия, повторы и второстепенные подробности.
4. Сформулируй итоговый ответ в требуемом формате.

Формат ответа:
Одно самостоятельное предложение без заголовка, списка и вступления.
""".strip()

class SupportServiceError(RuntimeError):
    pass

@dataclass(frozen=True)
class SupportSummary:
    text: str
    model: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str

class SupportService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def summarize(self, user_text: str) -> SupportSummary:
        text = user_text.strip()
        if not text:
            raise SupportServiceError("Обращение не должно быть пустым.")

        started_at = perf_counter()
        try:
            llm_result = self._llm_client.generate(self._build_messages(text))
        except LLMClientError as error:
            raise SupportServiceError(
                f"Не удалось получить ответ модели: {error}"
            ) from error
        elapsed_seconds = perf_counter() - started_at

        if llm_result.finish_reason == "length":
            raise SupportServiceError(
                "Ответ модели остановлен из-за ограничения длины."
            )
        if llm_result.finish_reason != "stop":
            raise SupportServiceError(
                "Модель не вернула готовое резюме. "
                f"Причина завершения: {llm_result.finish_reason}."
            )

        summary = self._validate_summary(llm_result.text)

        return SupportSummary(
            text=summary,
            model=llm_result.model,
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
                "role": "system",
                "content": DEVELOPER_INSTRUCTION ,
            },
            {
                "role": "user",
                "content": (
                    "<customer_request>\n"
                    f"{user_text}\n"
                    "</customer_request>"
                ),
            },
        ]

    @staticmethod
    def _validate_summary(summary: str | None) -> str:
        if summary is None:
            raise SupportServiceError("Модель не вернула текст резюме.")

        cleaned_summary = summary.strip()
        if not cleaned_summary:
            raise SupportServiceError("Модель вернула пустое резюме.")

        return cleaned_summary