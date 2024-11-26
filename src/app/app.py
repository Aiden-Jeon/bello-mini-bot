from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma

from agent.doc_agent import DocAgent
from settings import settings
from embedding import EMBEDDING_MODEL

st.title("🦜🔗 Langchain Quickstart App")

@st.cache_resource
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


db = load_vector_db()
retriever = db.as_retriever()
doc_agent = DocAgent(retriever)

with st.form("from"):
    user_input = st.text_area(
        "Enter text:",
        "What is this document is about?",
    )
    submitted = st.form_submit_button("Submit")


def preprocess(stream):
    for chunk in stream:
        if chunk == "<|end|>":
            continue
        else:
            yield chunk


output_container = st.empty()
if submitted:
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🦜")

    doc_agent.stream(user_input, answer_container)
