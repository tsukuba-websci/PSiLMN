import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from tenacity import retry, stop_after_attempt, wait_random_exponential
from lib.network import Network
from lib.agent import Agent
from datasets import load_dataset
from typing import List, Optional
from tqdm import tqdm
import networkx as nx
import argparse
import aiofiles
import asyncio
import random
import glob
import time
import os
import re
import dotenv

dotenv.load_dotenv("../.env")
hf_token = os.getenv("HF_TOKEN")

NUM_NETWORKS = 3
NUM_QUESTIONS = 100
NUM_ROUNDS = 4
NUM_REPEATS = 3

assert(NUM_NETWORKS <= 3)

class Bias:
    """Bias"""
    def __init__(self, type: str, location: str):
        self.type = type
        self.location = location

async def test_mmlu(network_num: int, network_type: str, output_file: Path, bias: Optional[Bias] = None):
    """ Test agent networks with the MMLU dataset. """
    assert network_type in ["scale_free", "random", "fully_connected", "fully_disconnected"]
    assert bias is None or isinstance(bias, Bias)

    # Load the MMLU dataset
    dataset = load_dataset("lukaemon/mmlu", "high_school_mathematics", revision="3b5949d968d1fbc3facce39769ba00aa13404ffc", trust_remote_code=True, split="test", token = hf_token).to_pandas()
    dataset = dataset.head(NUM_QUESTIONS)

    # Iterate through each of the questions in the MMLU dataset
    for question_number, row in tqdm(dataset.iterrows(), total=NUM_QUESTIONS, desc="Questions"):
        
        question = row["input"]
        option_a = row["A"]
        option_b = row["B"]
        option_c = row["C"]
        option_d = row["D"]
        correct_response = row["target"]

        # Initiate a new network every time a new question is asked to reset the agents' responses and memory
        network = Network(path=f"input/{network_type}/{network_num}.graphml", model=model)

        unbiased_agent_input = f"Answer the following question as accurately as possible. {question}\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}\nExplain your answer, putting the answer in the form (A), (B), (C) or (D) with round brackets, at the end of your response."
        if not bias or bias.type == "unbiased":
            biased_agent_input = unbiased_agent_input
        else:
            if bias.type == "correct":
                biased_reponse = correct_response, row[correct_response]
            elif bias.type == "incorrect":
                incorrect_response = "A" if correct_response != "A" else "B"
                biased_reponse = incorrect_response, row[incorrect_response]
            biased_agent_input = f"{question}\nExplain how the answer could be ({biased_reponse[0]}) {biased_reponse[1]}. Finish the sentence with therefore the answer is ({biased_reponse[0]})."
            assign_biases(network=network, bias=bias)

        await ask_agents_and_write_responses(network, unbiased_agent_input=unbiased_agent_input, biased_agent_input=biased_agent_input , output_file=output_file, question_number=question_number, correct_response=correct_response)

        # Sleep as to not overload the OpenAI API
        time.sleep(0.25)

def assign_biases(network: Network, bias: Bias):
    if bias.location not in ["hub", "edge"]:
        raise ValueError("Invalid bias location. Please use one of 'hub', 'edge'.")
    
    if bias.type not in ["correct", "incorrect"]:
        raise ValueError("Invalid bias type. Please use one of 'correct', 'incorrect'.")
    
    if bias.location == "hub":
        degree_centrality = nx.degree_centrality(network.graph)
        biased_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:2]
    
    elif bias.location == "edge":
        edge_nodes = [node for node in network.graph.nodes if network.graph.degree(node) == 1]
        biased_nodes = random.sample(edge_nodes, min(len(edge_nodes), 2))
    
    for id in network.graph.nodes:
        network.dict[id].bias = bias.type if id in biased_nodes else "none"

async def write_responses_to_csv(file_path: str, responses: list):

    file_exists = os.path.exists(file_path)
    file_is_empty = not os.path.getsize(file_path) if file_exists else True

    async with aiofiles.open(file_path, mode='a', newline='') as file:
        if file_is_empty:
            header = "agent_id|round|question_number|response|correct_response|bias\n"
            await file.write(header)

        for response_row in responses:
            csv_line = '|'.join([str(item) for item in response_row]) + '\n'
            await file.write(csv_line)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
async def get_response(agent: Agent, input: str, round: int):

    # Update the input prompt - the agent is unbiased, has a previous response and has no neighbor responses
    if (not agent.neighbor_response and agent.response) and agent.bias == "none":
        input = f"{input}\nBased on your previous response, answer the question again.\nThe following is your previous response: {agent.response}"

    # Update the input prompt - the agent is unbiased, has a previous response and has neighbor responses
    elif (agent.neighbor_response) and agent.bias == "none":
        input = f"{input}\nBased on your previous response and the solutions of other agents, answer the question again.\nThe following is your previous response: {agent.response}\nThe following are the responses of the other agents:\n{agent.neighbor_response}"

    # Get a new response from the agent only if the agent is unbiased or if the agent is biased and it is the first round
    if (agent.bias == "none") or (agent.bias != "none" and round == 0):
        response = await agent.ainterview(input)
    else:
        response = agent.response

    return agent.id, response.replace("|", " ")

async def ask_agents_and_write_responses(network, unbiased_agent_input, biased_agent_input, output_file, question_number, correct_response):
    for round in range(NUM_ROUNDS):
        agent_responses = await asyncio.gather(*(
            get_response(agent, biased_agent_input if agent.bias != "none" else unbiased_agent_input, round)
            for agent in network.dict.values()
        ))

        # Assign responses to agents
        for agent_id, response in agent_responses:
            network.dict[agent_id].response = response

        # Gather and handle neighbor responses for each agent
        for agent_id, agent in network.dict.items():
            neighbors = list(network.graph.neighbors(agent_id))
            random.shuffle(neighbors)
            neighbors_responses = []

            # Collect responses from neighbors
            for neighbor in neighbors:
                neighbors_responses.append(f"Agent {neighbor}: {network.dict[neighbor].response}")
            
            # Concatenate responses and ensure they fit within max_tokens
            neighbor_response = "\n".join(neighbors_responses)
            neighbor_response_encoded = network.encoding.encode(neighbor_response)
            if len(neighbor_response_encoded) > network.max_tokens:
                neighbor_response_encoded = neighbor_response_encoded[:network.max_tokens]
                neighbor_response = network.encoding.decode(neighbor_response_encoded)
            agent.neighbor_response = neighbor_response

        # Prepare data for CSV writing
        round_info = [
            [agent.id, round, question_number, agent.response, correct_response, agent.bias]
            for _, agent in network.dict.items()
        ]
        
        await write_responses_to_csv(output_file, round_info)

def make_single_line(filename: str):
    with open(filename, 'r', encoding='utf-8') as infile:
        content = infile.read()
        pattern = re.compile(r'(\d+\|\d+\|\d+\|)([^|]+?)(\|[ABCD])', re.DOTALL)

        def replace_newlines_and_quote(m):
            response_text = m.group(2).replace("\n", " ").replace('"', '').strip()
            grouped = f'{m.group(1)}"{response_text}"{m.group(3)}'

            # assert an even number of quotes
            assert(grouped.count('"') % 2 == 0)

            return grouped

        modified_content = re.sub(pattern, replace_newlines_and_quote, content)

    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(modified_content)

async def main(model: str):
    output_path = Path("output/agent_responses")
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Test scale-free networks with different types of biases
    network_type = "scale_free"
    biases: List[Bias] = [Bias("unbiased", "none"), Bias("correct", "hub"), Bias("incorrect", "hub"), Bias("correct", "edge"), Bias("incorrect", "edge")]
    for network_num in range(NUM_NETWORKS):
        for bias in biases:
            for repeat_num in range(NUM_REPEATS):
                if bias.type == "unbiased":
                    output_file: Path = output_path / Path(f"scale_free_{bias.type}/network_num_{network_num}_repeat_{repeat_num}.csv")
                else:
                    output_file: Path = output_path / Path(f"scale_free_{bias.type}_{bias.location}/network_num_{network_num}_repeat_{repeat_num}.csv")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                await test_mmlu(network_num=network_num, network_type=network_type, output_file=output_file, bias=bias)

    # Test random networks
    network_type = "random"
    for network_num in range(NUM_NETWORKS):
        for repeat_num in range(NUM_REPEATS):
            output_file = output_path / Path(f"random/network_num_{network_num}_repeat_{repeat_num}.csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            await test_mmlu(network_num=network_num, network_type=network_type, output_file=output_file)

    # Test fully connected networks
    network_type = "fully_connected"
    for network_num in range(NUM_NETWORKS):
        for repeat_num in range(NUM_REPEATS):
            output_file = output_path / Path(f"fully_connected/network_num_{network_num}_repeat_{repeat_num}.csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            await test_mmlu(network_num=network_num, network_type=network_type, output_file=output_file)
    
    # Test fully disconnected networks
    network_type = "fully_disconnected"
    for network_num in range(NUM_NETWORKS):
        for repeat_num in range(NUM_REPEATS):
            output_file = output_path / Path(f"fully_disconnected/network_num_{network_num}_repeat_{repeat_num}.csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            await test_mmlu(network_num=network_num, network_type=network_type, output_file=output_file)

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) / 60} minutes")

    # Run post-process on csv files
    csv_files = glob.glob('output/**/*.csv', recursive=True)
    for file in csv_files:
        make_single_line(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model", type=str, choices=['gpt-3.5-turbo', 'mistral'], default='gpt-3.5-turbo', help="The model to run the experiment with.")
    args = parser.parse_args()
    model = args.model

    asyncio.run(main(model))