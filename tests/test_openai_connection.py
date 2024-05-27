import pytest
import dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from lorem_text import lorem
import tiktoken

dotenv.load_dotenv("../.env")

def test_openai_connection():
    prompt = PromptTemplate.from_template("What is the capital of {country}?")
    llm_instance = ChatOpenAI(
        model_name="gpt-3.5-turbo-0125",
        max_tokens=20,
        request_timeout=60,
        max_retries=2,
    )
    response = LLMChain(llm =llm_instance,  prompt=prompt).invoke({"country": "Ireland"})["text"]

    response = response.lower()

    assert "dublin" in response

def test_max_context_window():

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    max_tokens = 16385 # max tokens for gpt-3.5-turbo

    input_tokens = 16500
    llm_input: str = lorem.words(input_tokens)

    llm_input_encoded = encoding.encode(llm_input)

    # limit the length of the string to the max tokens
    llm_input_encoded = llm_input_encoded[:max_tokens-100]

    llm_input = encoding.decode(llm_input_encoded)

    prompt = PromptTemplate.from_template("{input}")
    llm_instance = ChatOpenAI(
        model_name="gpt-3.5-turbo-0125",
        max_tokens=20,
        request_timeout=60,
        max_retries=2,
    )

    response = LLMChain(llm =llm_instance,  prompt=prompt).invoke({"input": f"{llm_input}"})["text"]

    # assert the length of the input tokens is less than the max tokens
    assert len(llm_input_encoded) < max_tokens

    # assert response is a string
    assert isinstance(response, str)

