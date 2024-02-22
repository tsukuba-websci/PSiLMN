import re
from typing import Any, Dict, List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import dotenv
from lib.memory import Memory
from langchain_community.llms import Ollama
import re
from typing import Any, Dict, List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import dotenv
from lib.memory import Memory
from langchain_community.llms import Ollama

dotenv.load_dotenv(".env")

class Agent:
    """Generative Agent"""

    def __init__(self, id: int, name: str, age: int = 40, hobby: str = "Not Applicable", job: str = "Not Applicable", personality: str = "Not Applicable", model: str = "mistral:instruct") -> None:
        self.id = id
        self.name = name
        self.age = age
        self.job = job
        self.hobby = hobby
        self.personality = personality
        self.verbose = False

        self.status = f"Name: {name}, Age: {age}, Job: {job}, Hobby: {hobby}, Personality: {personality}"
        self.opinion = 0

        self.llm = Ollama(model=model, num_predict=100)
        self.sentiment_llm = Ollama(model="mistral:instruct", num_predict=2)
        self.memory = Memory(model=model)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
    
    def sentiment_chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.sentiment_llm, prompt=prompt, verbose=self.verbose)

    def interview(self, question: str, correspodee: str = "Interviewer") -> str:
        """Generate a response to a given prompt."""

        prompt = PromptTemplate.from_template(
            "Respond with what {agent_name} would say."
            + "\nThe following is {agent_name}'s status: {agent_status}"
            + "\nThe following are {agent_name}'s relevant memories: {relevant_memories}"
            + "\n{correspondee} said: {question}"
            + "\n{agent_name}'s response:"
        )

        relevant_memories_str = "\n".join(self.memory.get_relevant_memories(question))
        kwargs: Dict[str, Any] = dict(
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            question=question,
            agent_status=self.status,
            correspondee=correspodee,
        )

        llm_query = self.chain(prompt=prompt).invoke(kwargs)
        return llm_query['text'].strip()
    
    def _clean_response(self, text: str) -> str:
        return re.sub(f"^{self.name} ", "", text.strip()).strip()

    def sentiment_analysis(self, text: str) -> int:
        """Analyse the sentiment of a given text."""
        prompt = PromptTemplate.from_template(
            "What is the sentiment of the following text?"
            + "\n\n"
            + "{text}"
            + "\n"
            + "Respond with ONLY with a whole number between -5 (negative) and 5 (positive)."
        )
        
        kwargs: Dict[str, Any] = dict(
            text=text,
        )

        attempt = 0
        max_attempts = 5  # Set a maximum number of attempts to prevent infinite loops

        while attempt < max_attempts:
            sentiment = self.sentiment_chain(prompt=prompt).invoke(kwargs)
            print(f"sentiment raw: {sentiment}")

            result = sentiment.get("text", "")
            parsed_sentiment = re.findall(r'-?\b\d+\b', result)
            if parsed_sentiment:
                return int(parsed_sentiment[0])
            
            attempt += 1  # Increment attempt counter if no integer was found

        print("No integer found in sentiment analysis result after maximum attempts.")
        return None

    def get_opinion(self, topic: str) -> str:
        response = self.interview(f"What is your opinion on the topic of {topic}?")
        print("Response:")
        print(response)
        sentiment = self.sentiment_analysis(response)
        return sentiment

def communicate(caller: Agent, callee: Agent, topic: str, firstInteraction: bool = False) -> List[str]:
    """
    A function that runs a conversation between two agents.

    :param caller: The agent that initiates the conversation.
    :param callee: The agent that responds to the conversation.
    :param firstInteraction: A boolean that indicates whether this is the first interaction between the two agents.

    :return: A string that represents the conversation between the two agents.
    """

    print(f"\n{caller.name} is communicating with {callee.name} about the topic: {topic}.\n")

    if firstInteraction:
        print("\nThis is the first interaction between the two agents.\n")
        observation = f"{caller.name}: Nice to meet you, I'm {caller.name}."
    else:
        print("\nThis is not the first interaction between the two agents.\n")
        observation = caller.interview(f"{caller.name}, what would you say to {callee.name} about the topic of {topic}?")

    observation = observation.replace('"', '')
    caller.memory.add_memory(observation)
    callee.memory.add_memory(observation)
    print(observation)

    turn = 1
    conversation = [observation]

    while True:
        # Limit the amount of possible speaking turns
        if turn >= 2:
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


