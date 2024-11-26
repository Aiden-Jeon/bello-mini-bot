from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class PathSettings(BaseSettings):
    DATASET_PATH: str


class LiteLLMSettings(BaseSettings):
    OPENAI_API_BASE: str
    OPENAI_API_KEY: str
    MODEL: str
    MAX_TOKENS: str


class LangChainSettings(BaseSettings):
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_API_KEY: str


class Settings(
    PathSettings,
    LiteLLMSettings,
    LangChainSettings,
):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


load_dotenv(Path(__file__).parent / "secret.env")
settings = Settings()

print(settings)
