import re
from typing import Any, Dict, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import dotenv
from lib.memory import Memory
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from faker import Faker
import random

dotenv.load_dotenv(".env")

class Agent:
    """Generative Agent"""

    def __init__(self, id: str, name: str, personality: str = "Not Applicable", model: str = "mistral") -> None:

        if "mistral" in model:
            llm = Ollama(model="mistral:instruct")
        elif model == "phi":
            llm = Ollama(model="phi")
        elif "gpt-3.5-turbo" in model:
            llm = ChatOpenAI()
        else:
            raise ValueError(f"Unknown model: {model}")

        self.id = id
        self.name = name
        self.personality = personality
        self.verbose = False
        self.status = f"Name: {name}, Personality: {personality}"
        self.response = ""
        self.neighbor_resonse = ""
        self.llm = llm
        self.memory = Memory(model=model)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def interview(self, question: str, correspodee: str = "Interviewer") -> str:
        """Generate a response to a given prompt."""
        prompt = PromptTemplate.from_template(
            "{correspondee}: {question}"
            +"\n{agent_name}:"
        )

        kwargs: Dict[str, Any] = dict(
            agent_name=self.name,
            question=question,
            correspondee=correspodee,
        )

        response = self.chain(prompt=prompt).invoke(kwargs)["text"].strip()
        
        return response

def communicate(caller: Agent, callee: Agent, question: str, debate_rounds: int = 3) -> List[str]:
    """
    A function that runs a conversation between two agents.

    :param caller: The agent that initiates the conversation.
    :param callee: The agent that responds to the conversation.

    :return: A string that represents the conversation between the two agents.
    """

    observation = f"{caller.name}: What is your answer to {question} and why?"

    caller.memory.add_memory(observation)
    print(observation)

    turn = 1
    conversation = [observation]

    while True:
        # Limit the amount of possible speaking turns
        if turn >= debate_rounds:
            force_goodbye_statement = f"{caller.name}: Excuse me {callee.name}, I have to go now!"
            conversation.append(force_goodbye_statement)
            caller.memory.add_memory(force_goodbye_statement)
            callee.memory.add_memory(force_goodbye_statement)
            print(force_goodbye_statement)
            break
        else:
            observation = callee.interview(question=observation, correspodee=caller.name)
            observation = observation.replace('"', '')
            conversation.append(observation)
            caller.memory.add_memory(observation)
            callee.memory.add_memory(observation)
            print(observation)

            observation = caller.interview(question=observation, correspodee=callee.name)
            observation = observation.replace('"', '')
            conversation.append(observation)
            caller.memory.add_memory(observation)
            callee.memory.add_memory(observation)
            print(observation)
        turn += 1

    return conversation

def fake_hobby():
    hobbies = ["Reading", "Hiking", "Painting", "Cooking", "Gaming", "Traveling", "Photography", "Gardening", "Yoga", "Dancing"]
    return random.choice(hobbies)

def fake_name():
    fake = Faker()
    return fake.name()

def fake_job():
    fake = Faker()
    return fake.job()

def fake_age():
    return random.randint(18, 65)

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer(input_str):
    pattern = r'\(([a-zA-Z])\)'
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    return solution

def parse_response_mmlu(response: str) -> Optional[str]:
    """
    Parse the response for MMLU questions
    """

    answer = parse_answer(response)

    return answer

