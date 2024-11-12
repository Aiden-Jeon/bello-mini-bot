from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_chroma import Chroma

from app.settings import settings
from app.embeddings import OPENAI_EMBEDDINGS
from app.llms import GPT_3_5_MODEL


DB_PATH = Path(__file__).parents[2] / settings.VECTOR_DB_ARTIFACT_PATH
print(DB_PATH)

persist_db = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=OPENAI_EMBEDDINGS,
    collection_name="my_db",
)
if len(persist_db.get(limit=1)) > 1:
    print("successfully vector db has been loaded.")

retriever = persist_db.as_retriever()
system_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
            You are chatbot, a bot designed to provide friendly and warm responses to help users using our Runway platform based on the information provided below.
            If you find it difficult to answer despite having the information, respond with, "Sorry, I can't answer this question."
            \n\n
            {context}
            \n\n
            """,
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


DOC_RAG = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | system_prompt
    | GPT_3_5_MODEL
    | StrOutputParser()
)
