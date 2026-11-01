import os

from time import perf_counter

import openai
from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-3.5-flash"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2000

SYSTEM_INSTRUCTION = (
    "Ты помогаешь оператору службы поддержки. "
    "Кратко пересказывай обращение одним предложением. "
    "Используй только факты из обращения. "
    "Не придумывай суммы, даты, причины и действия. "
    "Если данных недостаточно, прямо сообщай об этом."
)

def print_usage(usage) -> None:
    if usage is None:
        print("Провайдер не вернул статистику токенов")
        return

    print(f"Входные токены: {usage.prompt_tokens}")
    print(f"Выходные токены: {usage.completion_tokens}")
    print(f"Всего токенов: {usage.total_tokens}")

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

def summarize_request(client: OpenAI, user_text: str) -> None:
    """Кратко пересказывает обращение и печатает метрики запроса."""
    text = user_text.strip()

    if not text:
        print('Ошибка: Обращение не должно быть пустым.')
        return

    started_at = perf_counter()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=build_messages(text),
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )

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
    choice = response.choices[0]
    if choice.finish_reason == "length":
        print(
            "Ответ модели остановлен из-за ограничения длины. "
            "Увеличьте MAX_OUTPUT_TOKENS или сократите задачу."
        )
        return
    if choice.finish_reason != "stop":
        print(
            "Модель не вернула готовое резюме. "
            f"Причина завершения: {choice.finish_reason}."
        )
        return
    try:
        answer = validate_summarize(choice.message.content)
    except ValueError as error:
        print(f"Некорректный ответ модели: {error}")
        return

    print("Результат:")
    print(answer)

    print("\nМетрики:")
    print(f"Модель: {response.model}")
    print(f"Завершение: {choice.finish_reason}")
    print(f"Время: {elapsed_seconds:.2f} с")

    if response.usage is not None:
        print_usage(response.usage)

    print(f"ID ответа: {response.id}")

def main() -> None:
    load_dotenv()

    token = os.getenv("LLM_API_KEY")
    if not token:
        raise SystemExit(
            "Не найдена переменная LLM_API_KEY. "
            "Проверьте файл .env в корне проекта."
        )

    client = OpenAI(base_url=BASE_URL, api_key=token)
    user_text = input("Введите текст обращения: ")
    summarize_request(client, user_text)


if __name__ == "__main__":
    main()
