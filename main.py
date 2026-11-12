from app.console import run_console
from app.llm.client import LLMClient
from pydantic import ValidationError
from app.config import Settings
from app.support_service import SupportService
from app.chat_service import ChatService

def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return


    llm_client = LLMClient(settings)
    chat_service = ChatService(llm_client)

    try:
        run_console(
            chat_service,
            app_env=settings.app_env,
            configured_model=settings.model,
        )
    finally:
        llm_client.close()


if __name__ == "__main__":
    main()
