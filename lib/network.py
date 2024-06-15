"""
    Contains the code to initiate a network of agents
"""

from typing import Any, Tuple, Dict
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
import networkx as nx
import tiktoken
from lib.agent import Agent

class Network:
    """Network of Agents with Memory"""

    def __init__(self, path: str, model: str = "mistral") -> None:
        if "mistral" in model:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            token_buffer = 1000
            max_tokens = 16385 - token_buffer # max tokens for gpt-3.5-turbo

        elif "gpt-3.5-turbo" in model:
            from transformers import AutoTokenizer
            encoding = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            token_buffer = 1000
            max_tokens = 8192 - token_buffer
    
        (graph, dict) = self._load_agents(model=model, graph_location=path)

        self.graph = graph
        self.dict = dict
        self.encoding = encoding
        self.max_tokens = max_tokens

    def _load_agents(self, model: str, graph_location: str) -> Tuple[nx.Graph, Dict[int, Agent]]:
        graph = nx.read_graphml(graph_location)
        dict = {}

        for id in graph.nodes:
            dict[id] = Agent(id=id, model=model)

        return graph, dict
