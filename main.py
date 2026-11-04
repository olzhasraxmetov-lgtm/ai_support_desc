from time import perf_counter

import openai
from app.llm.client import LLMClient, LLMResult
from pydantic import ValidationError
from app.config import Settings


SYSTEM_INSTRUCTION = (
    "Ты помогаешь оператору службы поддержки. "
    "Кратко пересказывай обращение одним предложением. "
    "Используй только факты из обращения. "
    "Не придумывай суммы, даты, причины и действия. "
    "Если данных недостаточно, прямо сообщай об этом."
)

def print_usage(result: LLMResult) -> None:
    if result.total_tokens is None:
        print("Провайдер не вернул статистику токенов")
        return

    print(f"Входные токены: {result.prompt_tokens}")
    print(f"Выходные токены: {result.completion_tokens}")
    print(f"Всего токенов: {result.total_tokens}")

def build_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": f"Обращение:\n{user_text}",
        }
    ]

def validate_summarize(summary: str | None) -> str:
    if summary is None:
        raise ValueError("Модель не вернула текст")
    cleaned = summary.strip()
    if not cleaned:
        raise ValueError("Модель вернула пустой ответ")
    if len(cleaned) > 300:
        raise ValueError("Резюме получилось слишком длинным")

    return cleaned

def summarize_request(client: LLMClient, user_text: str, settings: Settings) -> None:
    """Кратко пересказывает обращение и печатает метрики запроса."""
    text = user_text.strip()

    if not text:
        print('Ошибка: Обращение не должно быть пустым.')
        return

    started_at = perf_counter()

    try:
        result = client.generate(build_messages(text))

    except openai.AuthenticationError:
        print("Ошибка авторизации: проверьте LLM_API_KEY.")
        return
    except openai.PermissionDeniedError:
        print("Нет доступа к модели: проверьте разрешения API-ключа.")
        return
    except openai.RateLimitError:
        print("Достигнут лимит запросов провайдера. Повторите запрос позднее.")
        return
    except openai.APIConnectionError:
        print("Не удалось соединиться с Gemini API. Проверьте сеть.")
        return
    except openai.APIStatusError as error:
        if error.status_code == 429:
            print("Превышен лимит бесплатных запросов Gemini. Подождите и повторите.")
        elif error.status_code == 401:
            print("Неверный API-ключ. Проверьте GEMINI_API_KEY.")
        elif error.status_code == 404:
            print("Модель не найдена. Проверьте название в MODEL.")
        else:
            print(f"Ошибка API: {error.status_code}")
        return

    elapsed_seconds = perf_counter() - started_at

    if result.finish_reason == "length":
        print(
            "Ответ модели остановлен из-за ограничения длины: "
            f"{settings.max_output_tokens} токенов."
        )
        return
    if result.finish_reason != "stop":
        print(
            "Модель не вернула готовое резюме. "
            f"Причина завершения: {result.finish_reason}."
        )
        return
    try:
        answer = validate_summarize(result.text)
    except ValueError as error:
        print(f"Некорректный ответ модели: {error}")
        return

    print("Результат:")
    print(answer)

    print("\nМетрики:")
    print(f"Модель: {result.model}")
    print(f"Завершение: {result.finish_reason}")
    print(f"Время: {elapsed_seconds:.2f} с")

    print_usage(result)

    print(f"ID ответа: {result.response_id}")

def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return


    client = LLMClient(settings)
    print(f"Окружение: {settings.app_env}")
    print(f"Модель: {settings.model}")

    user_text = input("Введите текст обращения: ")
    summarize_request(client, user_text, settings)


if __name__ == "__main__":
    main()
