from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from model import LLM_MODEL
from thought import LLMThought


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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

SYSTEM_CHAIN = (
    {
        "input": lambda x: x["input"],
        "context": lambda x: _format_docs(x["context"]),
    }
    | system_prompt
    | LLM_MODEL
    | StrOutputParser()
)


def _preprocess(stream):
    for chunk in stream:
        if chunk == "<|end|>":
            continue
        else:
            yield chunk


class DocAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def stream(self, input_text, container):
        retrieved_docs = self.retriever.invoke(input_text)
        for doc in retrieved_docs:
            retriever_thought = LLMThought(container)
            retriever_thought.search(doc)
        response = SYSTEM_CHAIN.stream({"input": input_text, "context": retrieved_docs})
        container.write_stream(_preprocess(response))
