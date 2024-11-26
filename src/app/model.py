from langchain_openai import ChatOpenAI
from settings import settings


LLM_MODEL = ChatOpenAI(
    openai_api_base=settings.OPENAI_API_BASE,
    model=settings.MODEL,
    openai_api_key=settings.OPENAI_API_KEY,
    max_tokens=int(settings.MAX_TOKENS),
)
