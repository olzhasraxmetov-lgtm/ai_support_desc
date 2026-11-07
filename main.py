from app.console import run_console
from app.llm.client import LLMClient
from pydantic import ValidationError
from app.config import Settings
from app.support_service import SupportService

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
    support_service = SupportService(llm_client)

    try:
        run_console(
            support_service,
            app_env=settings.app_env,
            configured_model=settings.model,
        )
    finally:
        llm_client.close()


if __name__ == "__main__":
    main()
