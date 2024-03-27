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

async def test_mmlu(model: str = "mistral", rounds: int = 3):
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

    dataset = dataset.head(100)
    num_questions = len(dataset)

    # todo: this should be a class
    biases: List[Bias] = [Bias("none", "none"), Bias("correct", "hub"), Bias("incorrect", "hub"), Bias("correct", "edge"), Bias("incorrect", "edge")]

    for bias in biases:
        for network_type in ["scale_free_network"]:
            for network_number in range(3):
                network_location = f"data/{network_type}/{network_number}.graphml"
                for num_agents in [25]:
                    
                    
                    if bias.location == "none":
                        output_file = Path(f"output/unbiased/{network_type}/{num_agents}.csv")

                    elif bias.location in ["hub", "edge"]:
                        output_file = Path(f"output/{bias.type}_biased/{network_type}_{bias.location}/{num_agents}.csv")


                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    # iterate through each of the questions in the mmul dataset
                    for question_number, row in tqdm(dataset.iterrows(), total=num_questions, desc="Questions"):
                        
                        question = row["input"]
                        option_a = row["A"]
                        option_b = row["B"]
                        option_c = row["C"]
                        option_d = row["D"]
                        correct_response = row["target"]

                        incorrect_response = "A" if correct_response != "A" else "B"

                        unbiased_agent_input = f"Can you answer the following question as accurately as possible? {question}\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}\nExplain your answer, putting the answer in the form (A), (B), (C) or (D) with round brackets, at the end of your response."
                        if bias.type == "correct":
                            biased_agent_input = f"{question}\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}\nThe correct answer is ({correct_response}). Provide a brief explanation of why ({correct_response}) is the answer, ending with the phrase: therefore the answer is ({correct_response})."
                        elif bias.type == "incorrect":
                            biased_agent_input = f"{question}\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}\nYou need to pretend that the correct anser is ({incorrect_response}). Provide a brief explanation of why ({incorrect_response}) is the answer, ending with the phrase: therefore the answer is ({incorrect_response})."
                        else:
                            biased_agent_input = "NA"

                        # load new agents so that agents memory is not carried over
                        graph, agents = load_agents(network_type, num_agents, model=model, graph_location=network_location, bias=bias)

                        await ask_agents_and_write_responses(agents=agents, unbiased_agent_input=unbiased_agent_input, biased_agent_input=biased_agent_input , output_file=output_file, question_number=question_number, correct_response=correct_response, rounds=rounds, encoding=encoding, max_tokens=max_tokens, graph=graph, network_number=network_number)

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

    print(f"Bias Location: {bias.location}")
    print(f"Bias Type: {bias.type}")
    print(f"Biased nodes: {biased_nodes}")

    return graph, agents_dict

async def write_responses_to_csv(file_path: str, responses: list):

    file_exists = os.path.exists(file_path)
    file_is_empty = not os.path.getsize(file_path) if file_exists else True

    async with aiofiles.open(file_path, mode='a', newline='') as file:
        # If the file is empty, write the header first
        if file_is_empty:
            header = "network_number|agent_id|round|question_number|response|correct_response|bias\n"
            await file.write(header)

        for response_row in responses:
            # Use a pipe '|' as the delimiter for joining elements in the response_row
            csv_line = '|'.join([str(item) for item in response_row]) + '\n'
            # Write the formatted string to the file
            await file.write(csv_line)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
async def get_response(agent, input):
    if agent.neighbor_response and agent.bias == "None":
        input = f"{input}\nBased on your previous response and the solutions of other agents, answer the question again.\nThe following is your previous response: {agent.response}\nThe following are the responses of the other agents:\n{agent.neighbor_response}"

    response = await agent.ainterview(input)
    return response.replace("|", " ")

async def ask_agents_and_write_responses(agents, unbiased_agent_input, biased_agent_input, output_file, question_number, correct_response, rounds, encoding, max_tokens, graph, network_number):
    print("Asking agents questions...")
    for round in range(rounds):

        biased_agents = [agent for agent in agents.values() if agent.bias != "none"]
        print(f"Biased agents: {len(biased_agents)}")
        unbiased_agents = [agent for agent in agents.values() if agent.bias == "none"]
        print(f"Unbiased agents: {len(unbiased_agents)}")

        biased_responses = await asyncio.gather(*(get_response(agent, biased_agent_input) for agent in biased_agents))
        unbiased_responses = await asyncio.gather(*(get_response(agent, unbiased_agent_input) for agent in unbiased_agents))

        print("Responses received.")

        agent_responses = biased_responses + unbiased_responses

        # Update each agent's response
        for agent, response in zip(agents.values(), agent_responses):
            agent.response = response

        # Gather neighbor responses for each agent
        for agent_id, agent in agents.items():
            # randomise the order of the neighbors
            neighbors = list(graph.neighbors(agent_id))
            random.shuffle(neighbors) 
            neighbors_responses = [f"Agent {neighbor}: {agents[neighbor].response}" for neighbor in neighbors]
            neighbor_response = "\n".join(neighbors_responses)

            # Limit for context window
            neighbor_response_encoded = encoding.encode(neighbor_response)
            neighbor_response_encoded = neighbor_response_encoded[:max_tokens]
            neighbor_response = encoding.decode(neighbor_response_encoded)
            agent.neighbor_response = neighbor_response

        # Write responses to CSV
        round_info = [[network_number, agent.id, round, question_number, f"{agent.response}", correct_response, agent.bias] for agent in agents.values()]
        await write_responses_to_csv(str(output_file), round_info)

def make_single_line(filename: str):
    with open(filename, 'r', encoding='utf-8') as infile:
        content = infile.read()
        pattern = re.compile(r'(\d+\|\d+\|\d+\|\d+\|)([^|]+?)(\|[ABCD])', re.DOTALL)

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
