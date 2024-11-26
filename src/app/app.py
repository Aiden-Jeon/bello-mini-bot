import streamlit as st

from agent.doc_agent import DOC_RAG_CHAIN

st.title("🦜🔗 Langchain Quickstart App")

with st.form("from"):
    user_input = st.text_area(
        "Enter text:",
        "Which panel has the highest average temperature among all panels?",
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

    # get data from vector database
    extracted_data = sql_agent.invoke(user_input, answer_container)
    # generate answer from rag
    explain_stream = DOC_RAG_CHAIN.stream(
        {"table": extracted_data, "question": user_input}
    )

    answer_container.write_stream(preprocess(explain_stream))
