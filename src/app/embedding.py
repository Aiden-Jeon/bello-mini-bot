from langchain_openai import OpenAIEmbeddings
from settings import settings


EMBEDDING_MODEL = OpenAIEmbeddings(
    openai_api_base=settings.OPENAI_API_BASE,
    openai_api_key=settings.OPENAI_API_KEY,
)
