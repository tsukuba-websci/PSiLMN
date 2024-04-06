import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from tenacity import retry, stop_after_attempt, wait_random_exponential
from lib.agent import Agent
from lib.bias import Bias
from datasets import load_dataset
from typing import Dict, Tuple, List
from tqdm import tqdm
import networkx as nx
import argparse
import tiktoken
import aiofiles
import asyncio
import random
import glob
import time
import os
import re

NUM_NETWORKS = 3
NUM_QUESTIONS = 100
NUM_ROUNDS = 4
NUM_REPEATS = 3

assert(NUM_NETWORKS <= 3)

async def test_mmlu(model: str = "mistral"):
    """
    Test agent networks with the MMLU dataset.

    Args:
        model (str): The model to run the experiment with. Should be one of 'mistral', 'phi', or 'gpt-3.5-turbo'.
        rounds (int): The number of rounds to run the experiment for.
    """
    
    if "gpt-3.5-turbo" in model:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_buffer = 1000
        max_tokens = 16385 - token_buffer # max tokens for gpt-3.5-turbo

    elif "mistral" in model:
        from transformers import AutoTokenizer
        encoding = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        token_buffer = 1000
        max_tokens = 8192 - token_buffer

    dataset = load_dataset("lukaemon/mmlu", "high_school_mathematics", revision="3b5949d968d1fbc3facce39769ba00aa13404ffc", trust_remote_code=True, split="test").to_pandas()

    dataset = dataset.head(NUM_QUESTIONS)

    biases: List[Bias] = [Bias("none", "none"), Bias("correct", "hub"), Bias("incorrect", "hub"), Bias("correct", "edge"), Bias("incorrect", "edge")]

    for bias in biases:
        for network_type in ["scale_free_network"]:
            for network_number in range(NUM_NETWORKS):
                network_location = f"data/{network_type}/{network_number}.graphml"
                for num_agents in [25]:
                    for repeat in range(NUM_REPEATS):
                    
                        if bias.location == "none":
                            output_file = Path(f"output/unbiased/agent_responses/{network_type}/{num_agents}.csv")

                        elif bias.location in ["hub", "edge"]:
                            output_file = Path(f"output/{bias.type}_biased/agent_responses/{network_type}_{bias.location}/{num_agents}.csv")


                        output_file.parent.mkdir(parents=True, exist_ok=True)

                        # iterate through each of the questions in the mmul dataset
                        for question_number, row in tqdm(dataset.iterrows(), total=NUM_QUESTIONS, desc="Questions"):
                            
                            question = row["input"]
                            option_a = row["A"]
                            option_b = row["B"]
                            option_c = row["C"]
                            option_d = row["D"]
                            correct_response = row["target"]

                            unbiased_agent_input = f"Answer the following question as accurately as possible. {question}\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}\nExplain your answer, putting the answer in the form (A), (B), (C) or (D) with round brackets, at the end of your response."

                            if bias.type == "none":
                                # the biased response is unused so could be anything
                                biased_agent_input = unbiased_agent_input
                            else:
                                if bias.type == "correct":
                                    biased_reponse = correct_response, row[correct_response]
                                elif bias.type == "incorrect":
                                    incorrect_response = "A" if correct_response != "A" else "B"
                                    biased_reponse = incorrect_response, row[incorrect_response]

                                biased_agent_input = f"{question}\nExplain how the answer could be ({biased_reponse[0]}) {biased_reponse[1]}. Finish the sentence with therefore the answer is ({biased_reponse[0]})."

                            # load new agents so that agents memory is not carried over
                            graph, agents = load_agents(network_type, num_agents, model=model, graph_location=network_location, bias=bias)

                            await ask_agents_and_write_responses(agents=agents, unbiased_agent_input=unbiased_agent_input, biased_agent_input=biased_agent_input , output_file=output_file, question_number=question_number, correct_response=correct_response, encoding=encoding, max_tokens=max_tokens, graph=graph, network_number=network_number, repeat=repeat)

                            time.sleep(5)

def load_agents(network_type: str, n: int, model: str, graph_location: str, bias: Bias) -> Tuple[nx.Graph, Dict[int, Agent]]:
    """
    Generate a scale free network of n nodes, where each node is an agent.
    
    Args:
        network_type (str): The type of network to generate. Should be one of 'scale_free_network', 'watts_strogatz_network', 'random_network', 'fully_connected_network', 'fully_disconnected_network'.
        n (int): The number of nodes in the network. Should be one of 5, 10, 25, 50, 100.
        model (str): The model to run the experiment with. Should be one of 'mistral', 'phi', or 'gpt-3.5-turbo'.
    """
    if bias.location not in ["hub", "edge", "none"]:
        raise ValueError("Invalid bias location. Please use one of 'hub', 'edge', 'none'.")
    
    if bias.type not in ["correct", "incorrect", "none"]:
        raise ValueError("Invalid bias type. Please use one of 'correct', 'incorrect', 'none'.")

    if n not in [5, 10, 25, 50, 100] or network_type not in ["scale_free_network", "watts_strogatz_network", "random_network", "fully_connected_network", "fully_disconnected_network"]:
        raise ValueError("Invalid network size or type. Please use one of 5, 10, 25, 100, 1000 for the network size and one of scale_free_network, watts_strogatz_network, random_network, fully_connected_network, fully_disconnected_network for the network type.")
    else:
        graph =  nx.read_graphml(graph_location)

    agents_dict = {}
    biased_nodes = []

    # Assign bias to nodes based
    if bias.location == "hub":
        degree_centrality = nx.degree_centrality(graph)
        biased_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:2]

    elif bias.location == "edge":
        edge_nodes = [node for node in graph.nodes if graph.degree(node) == 1]
        biased_nodes = random.sample(edge_nodes, min(len(edge_nodes), 2))  # Ensure we don't exceed 2 biased nodes

    for id in graph.nodes:
        agents_dict[id] = Agent(id=id, model=model, bias=bias.type if id in biased_nodes else "none")

    return graph, agents_dict

async def write_responses_to_csv(file_path: str, responses: list):

    file_exists = os.path.exists(file_path)
    file_is_empty = not os.path.getsize(file_path) if file_exists else True

    async with aiofiles.open(file_path, mode='a', newline='') as file:
        # If the file is empty, write the header first
        if file_is_empty:
            header = "network_number|agent_id|round|question_number|repeat|response|correct_response|bias\n"
            await file.write(header)

        for response_row in responses:
            # Use a pipe '|' as the delimiter for joining elements in the response_row
            csv_line = '|'.join([str(item) for item in response_row]) + '\n'
            # Write the formatted string to the file
            await file.write(csv_line)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
async def get_response(agent: Agent, input: str, round: int):

    # Alter the agents input if the agent is unbiased and has received a response from a neighbor
    if agent.neighbor_response and agent.bias == "none":
        input = f"{input}\nBased on your previous response and the solutions of other agents, answer the question again.\nThe following is your previous response: {agent.response}\nThe following are the responses of the other agents:\n{agent.neighbor_response}"
    
    # Get a new response from the agent only if the agent is unbiased or if the agent is biased and it is the first round
    if (agent.bias == "none") or (agent.bias != "none" and round == 0):
        response = await agent.ainterview(input)
    else:
        response = agent.response

    return agent.id, response.replace("|", " ")

async def ask_agents_and_write_responses(agents, unbiased_agent_input, biased_agent_input, output_file, question_number, correct_response, encoding, max_tokens, graph, network_number, repeat):
    for round in range(NUM_ROUNDS):
        agent_responses = await asyncio.gather(*(
            get_response(agent, biased_agent_input if agent.bias != "none" else unbiased_agent_input, round)
            for agent in agents.values()
        ))

        # Assign responses to agents
        for agent_id, response in agent_responses:
            agents[agent_id].response = response

        # Gather and handle neighbor responses for each agent
        for agent_id, agent in agents.items():
            neighbors = list(graph.neighbors(agent_id))
            random.shuffle(neighbors)
            neighbors_responses = []

            # Collect responses from neighbors
            for neighbor in neighbors:
                neighbors_responses.append(f"Agent {neighbor}: {agents[neighbor].response}")
            
            # Concatenate responses and ensure they fit within max_tokens
            neighbor_response = "\n".join(neighbors_responses)
            neighbor_response_encoded = encoding.encode(neighbor_response)
            if len(neighbor_response_encoded) > max_tokens:
                neighbor_response_encoded = neighbor_response_encoded[:max_tokens]
                neighbor_response = encoding.decode(neighbor_response_encoded)
            agent.neighbor_response = neighbor_response

        # Prepare data for CSV writing
        round_info = [
            [network_number, agent.id, round, question_number, repeat, agent.response, correct_response, agent.bias]
            for _, agent in agents.items()
        ]
        
        await write_responses_to_csv(output_file, round_info)

def make_single_line(filename: str):
    with open(filename, 'r', encoding='utf-8') as infile:
        content = infile.read()
        pattern = re.compile(r'(\d+\|\d+\|\d+\|\d+\|\d+\|)([^|]+?)(\|[ABCD])', re.DOTALL)

        def replace_newlines_and_quote(m):
            response_text = m.group(2).replace("\n", " ").replace('"', '').strip()
            grouped = f'{m.group(1)}"{response_text}"{m.group(3)}'

            # assert an even number of quotes
            assert(grouped.count('"') % 2 == 0)

            return grouped

        modified_content = re.sub(pattern, replace_newlines_and_quote, content)

    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(modified_content)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("model", type=str, choices=['gpt-3.5-turbo', 'mistral'], default='gpt-3.5-turbo', help="The model to run the experiment with.")
    args = parser.parse_args()
    model = args.model

    start_time = time.time()

    # test mmlu
    asyncio.run(test_mmlu(model=model))

    end_time = time.time()

    print(f"Time taken: {(end_time - start_time) / 60} minutes")

    # # run post process on csv files
    csv_files = glob.glob('output/**/*.csv', recursive=True)
    for file in csv_files:
        make_single_line(file)

    pass
