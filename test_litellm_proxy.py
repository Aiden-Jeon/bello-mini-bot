import re

import pandas as pd
from langchain_openai import ChatOpenAI

#
# LLM Model
#
model_list = {
    "phi-3.5": ChatOpenAI(
        openai_api_base="http://0.0.0.0:4000",
        model="huggingface/microsoft/Phi-3.5-mini-instruct",
        openai_api_key="sk-1234",
        max_tokens=1000,
    ),
    "meta-llama": ChatOpenAI(
        openai_api_base="http://0.0.0.0:4000",
        model="huggingface/meta-llama/Meta-Llama-3-70B-Instruct",
        openai_api_key="sk-1234",
        max_tokens=1000,
    ),
    "gpt-3.5": ChatOpenAI(
        openai_api_base="http://0.0.0.0:4000",
        model="openai/gpt-3.5-turbo",
        openai_api_key="sk-1234",
    ),
    "gpt-4.0": ChatOpenAI(
        openai_api_base="http://0.0.0.0:4000",
        model="openai/gpt-4o-mini",
        openai_api_key="sk-1234",
    ),
}

#
# Test
#

question = "Who are you?"

for model_name, model in model_list.items():
    response = model.invoke(question)
    print(model_name)
    print(response.content)
    print("-" * 50)
