import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from lib.agent import Agent, fake_name
from datasets import load_dataset
from typing import Dict, Tuple
from tqdm import tqdm
import networkx as nx
import argparse
import tiktoken
import aiofiles
import asyncio
import random
import glob
import os
import re

async def test_mmlu_with_bias(model: str = "mistral", rounds: int = 3):
    """
    Test agent networks with the MMLU dataset.

    Args:
        model (str): The model to run the experiment with. Should be one of 'mistral', 'phi', or 'gpt-3.5-turbo'.
        rounds (int): The number of rounds to run the experiment for.
    """

    if "gpt-3.5-turbo" in model:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_buffer = 500
        max_tokens = 16385 - token_buffer
    
    dataset = load_dataset("lukaemon/mmlu", "high_school_mathematics", revision="3b5949d968d1fbc3facce39769ba00aa13404ffc", trust_remote_code=True, split="test").to_pandas()

    dataset = dataset.head(100)
    num_questions = len(dataset)

    for bias in ["correct", "incorrect"]:
        for network_type in ["scale_free_network"]:
            for num_agents in [10, 25, 50, 100]:
                agent_output_file = Path(f"output/biased/{bias}/agent_responses/{network_type}/{num_agents}.csv")
                agent_output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(agent_output_file, 'w') as file:
                    pass

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
                    if bias == "correct":
                        biased_agent_input = f"{question}\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}\nThe correct answer is ({correct_response}). Provide a brief explanation of why ({correct_response}) is the answer, ending with the phrase: therefore the answer is ({correct_response})."
                    elif bias == "incorrect":
                        biased_agent_input = f"{question}\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}\nYou need to pretend that the correct anser is ({incorrect_response}). Provide a brief explanation of why ({incorrect_response}) is the answer, ending with the phrase: therefore the answer is ({incorrect_response})."

                    # load new agents so that agents memory is not carried over
                    graph, agents = load_agents(network_type, num_agents, model=model, bias=bias)

                    await ask_agents_and_write_responses(agents=agents, unbiased_agent_input=unbiased_agent_input, biased_agent_input=biased_agent_input , agent_output_file=agent_output_file, question_number=question_number, correct_response=correct_response, rounds=rounds, encoding=encoding, max_tokens=max_tokens, graph=graph)

def load_agents(network_type: str, n: int, model: str, bias: str) -> Tuple[nx.Graph, Dict[int, Agent]]:
    """
    Generate a scale free network of n nodes, where each node is an agent.
    
    Args:
        network_type (str): The type of network to generate. Should be one of 'scale_free_network', 'watts_strogatz_network', 'random_network', 'fully_connected_network', 'fully_disconnected_network'.
        n (int): The number of nodes in the network. Should be one of 5, 10, 25, 50, 100.
        model (str): The model to run the experiment with. Should be one of 'mistral', 'phi', or 'gpt-3.5-turbo'.
    """

    if n not in [5, 10, 25, 50, 100] or network_type not in ["scale_free_network", "watts_strogatz_network", "random_network", "fully_connected_network", "fully_disconnected_network"]:
        raise ValueError("Invalid network size or type. Please use one of 5, 10, 25, 100, 1000 for the network size and one of scale_free_network, watts_strogatz_network, random_network, fully_connected_network, fully_disconnected_network for the network type.")
    else:
        graph =  nx.read_graphml(f"data/{network_type}/{n}.graphml")


        degree_centrality = nx.degree_centrality(graph)

        hub_threshold = 0.5  # Example threshold
        hub_nodes = [node for node, centrality in degree_centrality.items() if centrality > hub_threshold]

        agents_dict = {}
        for id in graph.nodes:
            agents_dict[id] = Agent(id=id, name=fake_name(), model=model, bias=bias if id in hub_nodes else "None")

        return graph, agents_dict

async def get_response(agent: Agent, input: str) -> str:

    if agent.neighbor_response and agent.bias == "None":
        input = f"{input}\nBased on your previous response and the solutions of other the agents, answer the question again.\nThe following is your previous response: {agent.response}\nThe following are the responses of the other agents:\n{agent.neighbor_response}"

    print(f"INPUT: {input}")

    response = await agent.ainterview(input)
    return response.replace("|", " ")

async def ask_agents_and_write_responses(agents, unbiased_agent_input, biased_agent_input, agent_output_file, question_number, correct_response, rounds, encoding, max_tokens, graph):
    for round in range(rounds):

        biased_agents = [agent for agent in agents.values() if agent.bias != "None"]
        unbiased_agents = [agent for agent in agents.values() if agent.bias == "None"]

        biased_responses = await asyncio.gather(*(get_response(agent, biased_agent_input) for agent in biased_agents))
        unbiased_responses = await asyncio.gather(*(get_response(agent, unbiased_agent_input) for agent in unbiased_agents))

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
        round_info = [[agent.id, round, question_number, f"{agent.response}", correct_response, agent.bias] for agent in agents.values()]
        await write_responses_to_csv(str(agent_output_file), round_info)

async def write_responses_to_csv(file_path: str, responses: list):

    file_exists = os.path.exists(file_path)
    file_is_empty = not os.path.getsize(file_path) if file_exists else True

    async with aiofiles.open(file_path, mode='a', newline='') as file:
        # If the file is empty, write the header first
        if file_is_empty:
            header = "agent_id|round|question_number|response|correct_response|bias\n"
            await file.write(header)

        for response_row in responses:
            # Use a pipe '|' as the delimiter for joining elements in the response_row
            csv_line = '|'.join([str(item) for item in response_row]) + '\n'
            # Write the formatted string to the file
            await file.write(csv_line)

def make_single_line(filename: str):
    with open(filename, 'r', encoding='utf-8') as infile:
        content = infile.read()
        pattern = re.compile(r'(\d+\|\d+\|\d+\|)([^|]+?)(\|[ABCD])', re.DOTALL)

        def replace_newlines_and_quote(m):
            response_text = m.group(2).replace("\n", " ").strip()
            return f'{m.group(1)}"{response_text}"{m.group(3)}'

        modified_content = re.sub(pattern, replace_newlines_and_quote, content)

    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(modified_content)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("model", type=str, choices=['gpt-3.5-turbo'], default='gpt-3.5-turbo', help="The model to run the experiment with.")
    args = parser.parse_args()
    model = args.model

    # # test mmlu
    asyncio.run(test_mmlu_with_bias(model=model))

    # run post process on csv files
    csv_files = glob.glob('output/biased/**/**/*.csv', recursive=True)
    for file in csv_files:
        make_single_line(file)

    pass
