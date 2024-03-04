import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from lib.agent import Agent, fake_name
import networkx as nx
from datasets import load_dataset
from typing import Dict, Tuple
from tqdm import tqdm
import csv
import argparse

def test_mmlu(model: str = "mistral", rounds: int = 3):
    """
    Test networks with the MMLU dataset.

    Args:
        model (str): The model to run the experiment with. Should be one of 'mistral', 'phi', or 'gpt-3.5-turbo'.
    """
    
    dataset = load_dataset("lukaemon/mmlu", "high_school_mathematics", revision="3b5949d968d1fbc3facce39769ba00aa13404ffc", trust_remote_code=True, split="test").to_pandas()

    # todo: undo the limit of questions
    dataset = dataset.head(2)
    num_questions = len(dataset)

    for network_type in ["scale_free_network", "watts_strogatz_network", "random_network", "fully_connected_network"]:

        for num_agents in [10]: # todo: for 10, 100, 1000

            # Construct the path for the output file
            agent_output_file = Path(f"output/agent_responses/{network_type}_{num_agents}.csv")
            # Ensure the directory exists
            agent_output_file.parent.mkdir(parents=True, exist_ok=True)

            # iterate through each of the questions in the mmul dataset
            for question_number, row in tqdm(dataset.iterrows(), total=num_questions, desc="Questions"):
                
                question = row["input"]
                option_a = row["A"]
                option_b = row["B"]
                option_c = row["C"]
                option_d = row["D"]
                correct_response = row["target"]

                agent_input = f"Can you answer the following question as accurately as possible? {question}\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}\nExplain your answer, putting the answer in the form (A), (B), (C) or (D) with round brackets, at the end of your response."

                # load new agents so that agents memory is not carried over
                graph, agents = load_agents(network_type, num_agents, model=model)

                for round in tqdm(range(rounds), desc="Rounds of Communication"):

                    # get each agent's response
                    for agent in agents.values():
                        agent.response = get_response(agent=agent, input=agent_input)

                        with open(agent_output_file, mode='a', newline='') as file:
                            # Initialize csv.writer with the correct delimiter here
                            writer = csv.writer(file, delimiter='|')
                            
                            # Check if the file is empty to write headers
                            if file.tell() == 0:
                                writer.writerow(['agent_id', 'round', 'question_number', 'response', 'correct_response'])
                            
                            # Now, you can write rows without specifying the delimiter
                            writer.writerow([agent.id, round, question_number, agent.response, correct_response])

                    # get the neighbor responses
                    for node in graph.nodes:
                        agent = agents[node]
                        # form a string on all neighbors' responses
                        neighbors_responses = [f"Agent {neighbor}: {agents[neighbor].response}" for neighbor in graph.neighbors(node)]
                        agent.neighbor_resonse = "\n".join(neighbors_responses)

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
        input = f"{input}\nUsing the solutions from other agents as additional information, give an updated response. The follo|ing are the responses of the other agents:\n{agent.neighbor_resonse}"

    response = agent.interview(input).replace("|", " ")
    
    return response

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("model", type=str, choices=['mistral', 'phi', 'gpt-3.5-turbo'], default='gpt-3.5-turbo', help="The model to run the experiment with.")

    args = parser.parse_args()

    model = args.model

    test_mmlu(model=model, rounds = 2)

    pass

