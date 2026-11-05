from time import perf_counter

from app.console import run_console
from app.llm.client import LLMClient, LLMResult
from app.llm.errors import LLMClientError
from pydantic import ValidationError
from app.config import Settings
from app.support_service import SupportService

SYSTEM_INSTRUCTION = (
    "Ты помогаешь оператору службы поддержки. "
    "Кратко пересказывай обращение одним предложением. "
    "Используй только факты из обращения. "
    "Не придумывай суммы, даты, причины и действия. "
    "Если данных недостаточно, прямо сообщай об этом."
)


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

    run_console(
        support_service,
        app_env=settings.app_env,
        configured_model=settings.model,
    )


if __name__ == "__main__":
    main()
