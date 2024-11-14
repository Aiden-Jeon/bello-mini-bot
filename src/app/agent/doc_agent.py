from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_chroma import Chroma

from app.settings import settings
from app.embeddings import OPENAI_EMBEDDINGS
from app.llms import GPT_3_5_MODEL


def load_vector_db():
    DB_PATH = Path(__file__).parents[2] / settings.VECTOR_DB_ARTIFACT_PATH
    db = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=OPENAI_EMBEDDINGS,
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
    return {
        "context": retriever | _format_docs,
        "question": RunnablePassthrough(),
    }


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
                "{question}",
            ),
        ]
    )
    retriever_chain = load_retriever_chain()

    return retriever_chain | system_prompt | GPT_3_5_MODEL | StrOutputParser()


DOC_RAG_AGENT = load_doc_rag_agent()
