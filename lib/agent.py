import re
from typing import Any, Dict, List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import dotenv
from lib.memory import Memory
from langchain_community.llms import Ollama
from faker import Faker
import random

dotenv.load_dotenv(".env")

class Agent:
    """Generative Agent"""

    def __init__(self, id: str, name: str, age: int = 40, hobby: str = "Not Applicable", job: str = "Not Applicable", personality: str = "Not Applicable", model: str = "mistral:instruct") -> None:
        self.id = id
        self.name = name
        self.age = age
        self.job = job
        self.hobby = hobby
        self.personality = personality
        self.verbose = False
        self.status = f"Name: {name}, Age: {age}, Job: {job}, Hobby: {hobby}, Personality: {personality}"
        self.response = ""
        self.neighbor_resonse = ""
        self.llm = Ollama(model=model, num_predict=1)
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
            "Respond with what {agent_name} would say."
            + "\nThe following is {agent_name}'s status: {agent_status}"
            + "\nThe following are {agent_name}'s relevant memories: {relevant_memories}"
            + "\n{correspondee}: {question}"
            + "\n{agent_name}:"
        )

        relevant_memories_str = "\n".join(self.memory.get_relevant_memories(question))
        kwargs: Dict[str, Any] = dict(
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            question=question,
            agent_status=self.status,
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