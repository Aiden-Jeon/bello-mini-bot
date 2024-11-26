from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_chroma import Chroma

from settings import settings
from embedding import EMBEDDING_MODEL
from model import LLM_MODEL


from pydantic import BaseModel, Field


def load_vector_db():
    DB_PATH = Path(__file__).parents[2] / settings.VECTOR_DB_ARTIFACT_PATH
    db = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=EMBEDDING_MODEL,
        collection_name="my_db",
    )
    if len(db.get(limit=1)) > 1:
        print("successfully vector db has been loaded.")
    else:
        raise ValueError("vector db is not loaded properly.")
    return db


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_retriever_chain():
    db = load_vector_db()
    retriever = db.as_retriever()
    retrieve_docs = (lambda x: x["input"]) | retriever
    return retrieve_docs


def load_doc_rag_agent():
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
                "{input}",
            ),
        ]
    )

    # 단계 7: 체인 생성(Create Chain)
    rag_chain_from_docs = (
        {
            "input": lambda x: x["input"],
            "context": lambda x: _format_docs(x["context"]),
        }
        | system_prompt
        | LLM_MODEL
    )

    return rag_chain_from_docs


retrieve_docs = load_retriever_chain()
rag_chain_from_docs = load_doc_rag_agent()
DOC_RAG_CHAIN = RunnablePassthrough.assign(context=retrieve_docs).assign(
    answer=rag_chain_from_docs
)
