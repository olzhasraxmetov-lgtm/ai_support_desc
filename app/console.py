from app.chat_service import (
    ChatReply,
    ChatService,
    ChatServiceError,
)

try:
    import readline
except ImportError:
    pass


def print_usage(result: ChatReply) -> None:
    if result.total_tokens is None:
        print("Провайдер не вернул статистику токенов.")
        return

    print(f"Входные токены: {result.prompt_tokens}")
    print(f"Выходные токены: {result.completion_tokens}")
    print(f"Всего токенов: {result.total_tokens}")


def run_console(
        service: ChatService,
        app_env: str,
        configured_model: str,
) -> None:
    print(f"Окружение: {app_env}")
    print(f"Настроенная модель: {configured_model}")
    print("Для выхода введите /exit.")

    while True:
        user_text = input("\nВы: ")
        if user_text.strip() == "/exit":
            return

        try:
            result = service.reply(user_text)
        except ChatServiceError as error:
            print(f"Ошибка чата: {error}")
            continue

        print(f"Assistant: {result.text}")
        print(f"Промпт: {result.prompt_id}@{result.prompt_version}")
        print(f"Время ответа: {result.elapsed_seconds:.2f} с")
        print_usage(result)
        print(f"ID ответа: {result.response_id}")