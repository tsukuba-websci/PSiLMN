import pytest
import dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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
    print(response)

    response = response.lower()

    assert "dublin" in response