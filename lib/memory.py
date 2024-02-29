import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.utils import mock_now
import faiss
import math
from langchain_community.docstore import InMemoryDocstore
import dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings

dotenv.load_dotenv(".env")

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1.0 - score / math.sqrt(2)

class Memory:
    """Memory for the generative agent."""

    def __init__(self, model: str = "mistral") -> None:

        if "mistral" in model:
            embedding_fn = OllamaEmbeddings(model="mistral:instruct")
            embedding_size = 4096
        elif model == "phi":
            embedding_fn = OllamaEmbeddings(model="phi")
            embedding_size = 2560
        elif "gpt-3.5-turbo" in model:
            embedding_size = 1536
            embedding_fn = OpenAIEmbeddings(model="gpt-3.5-turbo-0125")
        else:
            raise ValueError(f"Unknown model: {model}")
        
        self.llm = Ollama(model=model, num_predict=150)
        self.vectorstore = FAISS(embedding_fn, faiss.IndexFlatL2(embedding_size), InMemoryDocstore({}), {})
        self.memory_stream: List[Document] = []
        self.add_memory_key: str = "add_memory"
        self.now_key: str = "now"

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def add_memory(self, memory_content: str, now: Optional[datetime] = None) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        document = Document(
            page_content=memory_content,
            metadata={
                # "id": id,
            },
        )
        result = self.vectorstore.add_documents([document], current_time=now)
        self.memory_stream.append(document)
        return result

    def fetch_memories(self, observation: str, now: Optional[datetime] = None) -> List[Document]:
        """Fetch related memories."""
        if now is not None:
            with mock_now(now):
                return self.get_relevant_documents(observation)
        else:
            return self.get_relevant_documents(observation)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}[{created_time}] {memory.page_content.strip()}"

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    def get_relevant_memories(self, query: str) -> List[Document]:
        results = self.vectorstore.similarity_search(
            query
        )

        memory_content = [mem.page_content for mem in results]
        return memory_content

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this model run to memory."""
        # TODO: fix the save memory key
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if mem:
            self.add_memory(mem, now=now)