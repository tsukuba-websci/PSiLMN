import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from lib.agent import Agent, fake_name, parse_response_mmlu
import networkx as nx
from datasets import load_dataset
from typing import Dict, Tuple
from collections import Counter
import logging
from tqdm import tqdm
import csv
import argparse

# Configure logging
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename='logs/logfile.log', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def test_mmlu(model: str = "mistral"):
    """
    Test the multi-agent response to the MMLU dataset.
    """

    logging.info("MMLU dataset loading")
    dataset = load_dataset("lukaemon/mmlu", "high_school_mathematics", split="test").to_pandas()
    logging.info("MMLU dataset loaded")

    Path("results").mkdir(parents=True, exist_ok=True)
    csv_file_path = Path("results/results_mmlu.csv")

    # todo: undo the limit of questions
    dataset = dataset.head(3)
    num_questions = len(dataset)

    for network_type in ["scale_free_network", "watts_strogatz_network", "random_network", "fully_connected_network"]:
        logging.info(f"Running test for {network_type} network.")

        for num_agents in [10]: # todo: for 10, 100, 1000
            logging.info(f"Running test for {num_agents} agents.")

            correct_responses = []

            # iterate through each of the questions in the mmul dataset
            for index, row in tqdm(dataset.iterrows(), total=num_questions, desc="Questions"):
                
                logging.info(f"Question {index+1} of {num_questions}")

                question = row["input"]
                option_a = row["A"]
                option_b = row["B"]
                option_c = row["C"]
                option_d = row["D"]
                correct_response = row["target"]

                agent_input = f"Can you answer the following question as accurately as possible? {question}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nExplain your answer, putting the answer in the form (X) at the end of your response"

                # load new agents so that agents memory is not carried over
                graph, agents = load_agents(network_type, num_agents, model=model)

                rounds = 2
                logging.info(f"Running test for {rounds} rounds of communication.")

                for round in tqdm(range(rounds), desc="Rounds of Communication"):

                    logging.info(f"Round {round} of {rounds} of communication")

                    logging.info("Getting each agents response")
                    # get each agent's response
                    for agent in agents.values():
                        agent.response = get_response(agent=agent, input=agent_input)
                        logging.info(f"Agent {agent.id} response: {agent.response}")

                    parsed_responses = [parse_response_mmlu(agent.response) for agent in agents.values()]
                    logging.info(f"Parsed responses: {parsed_responses}")
                    response_counter = Counter(parsed_responses)
                    most_common_response, _ = response_counter.most_common(1)[0]
                    logging.info(f"Most common response: {most_common_response}")

                    # get the neighbor responses
                    for node in graph.nodes:
                        agent = agents[node]
                        # form a string on all neighbors' responses
                        neighbors_responses = [f"Agent {neighbor}: {agents[neighbor].response}" for neighbor in graph.neighbors(node)]
                        agent.neighbor_resonse = "\n".join(neighbors_responses)
                        logging.info(f"Agent {agent.id} neighbor response: {agent.neighbor_resonse}")

                logging.info("After the final round of communication")
                logging.info(f"Most common response: {most_common_response}")
                logging.info(f"Correct response: {correct_response}")
                if most_common_response == correct_response:
                    correct_responses.append(True)
                    logging.info("Correct response")
                else:
                    correct_responses.append(False)
                    logging.info("Incorrect response")

            logging.info("Calculating fraction of correct responses")    
            frac_correct = sum(correct_responses) / len(correct_responses)
            print(f"Fraction of correct responses: {frac_correct}")
            logging.info(f"Fraction of correct responses: {frac_correct}")

            with csv_file_path.open(mode='a', newline='') as file:
                writer = csv.writer(file)
                # Check if the file is empty to write headers
                if file.tell() == 0:
                    writer.writerow(['network_type','network_size', 'rounds', 'fraction_correct'])
                writer.writerow([network_type, num_agents, rounds, frac_correct])

def load_agents(network_type: str, n: int, model: str) -> Tuple[nx.Graph, Dict[int, Agent]]:
    """
    Generate a scale free network of n nodes, where each node is an agent.
    
    Args:
    n (int): The number of nodes in the network. Should be one of 10, 100, or 1000.
    """

    if n not in [10, 100, 1000] or network_type not in ["scale_free_network", "watts_strogatz_network", "random_network", "fully_connected_network", "fully_disconnected_network"]:
        raise ValueError("Invalid network size or type. Please use one of 10, 100, 1000 for the network size and one of scale_free_network, watts_strogatz_network, random_network, fully_connected_network, fully_disconnected_network for the network type.")
    else:
        graph =  nx.read_graphml(f"data/{network_type}/{n}.graphml")
        agents_dict = {}
        for id in graph.nodes:
            agents_dict[id] = Agent(id=id, name=fake_name(), model=model)

        return graph, agents_dict

def get_response(agent: Agent, input: str) -> str:
    """
    Get the agents response to a question.

    Args:
        agent (Agent): The agent to get a response from.
    """

    if agent.neighbor_resonse:
        input = f"{input}\nUsing the solutions from other agents as additional information, give an updated response. The following are the responses of the other agents:\n{agent.neighbor_resonse}"

    logging.info(f"Agent input: {input}")

    response = agent.interview(input)
    
    return response

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("model", type=str, choices=['mistral', 'phi', 'gpt-3.5-turbo'], help="The model to run the experiment with.")

    args = parser.parse_args()

    model = args.model

    logging.info("Starting test_mmlu with model: {model}")
    test_mmlu(model=model)

    pass

