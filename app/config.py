from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    llm_api_key: SecretStr = Field(
        validation_alias="LLM_API_KEY",
    )
    base_url: AnyHttpUrl = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        validation_alias="BASE_URL",
    )
    model: str = Field(
        default="gemini-3.5-flash",
        min_length=1,
        validation_alias="MODEL",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        validation_alias="TEMPERATURE",
    )
    max_output_tokens: int = Field(
        default=2000,
        gt=0,
        validation_alias="MAX_OUTPUT_TOKENS",
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        validation_alias="REQUEST_TIMEOUT_SECONDS",
    )
    app_env: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias="APP_ENV",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding = "utf-8",
        env_ignore_empty = True,
        extra = "ignore",
    )