import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from lib.agent import Agent, communicate, fake_name, fake_age, fake_job, fake_hobby
import networkx as nx
from datasets import load_dataset
from typing import Dict, Tuple
from collections import Counter

def test_mmlu():
    """
    Test the multi-agent response to the MMUL dataset.
    """
    
    dataset = load_dataset("lukaemon/mmlu", "sociology", split="test").to_pandas()

    # todo: undo the limit of questions
    dataset = dataset.head(10)

    for num_agents in [10]: # todo: for 10, 100, 1000

        correct_responses = []

        # iterate through each of the questions in the mmul dataset
        for index, row in dataset.iterrows():
            question = row["input"]
            correct_response = row["target"]
            valid_responses = ["A", "B", "C", "D"]

            agent_input = f"{question}\nA: {row["A"]}\nB: {row["B"]}\nC: {row["C"]}\nD: {row["D"]}\nRespond with only A, B, C, or D. Do not reply with Based, you must only say A, B, C, or D."
                
            # load new agents
            graph, agents = load_agents(num_agents)

            responses_list = []

            # single agent response
            for id, agent in agents.items():

                # Get the agents response the question
                response = agent.interview(agent_input).strip().upper()

                # Check if the response is valid, if not, try again 3 times
                try_counter = 0
                while response not in valid_responses and try_counter < 3:
                    response = agents[id].interview(agent_input)
                    try_counter += 1

                responses_list.append(response)
                responses_counter = Counter(responses_list)
                most_common_response, count = responses_counter.most_common(1)[0]


                if most_common_response == correct_response:
                    correct_responses.append(True)
                else:
                    correct_responses.append(False)

            # todo: multi-agent response
            # communicated_this_step = set()
            # for node in agents.keys():
            #     if node not in communicated_this_step:
            #         communicated_this_step.add(node)
            #         for neighbor in nx.neighbors(graph, node):
            #             if neighbor not in communicated_this_step:
            #                 communicate(agents[node], agents[neighbor])
                        
        num_correct_responses = correct_responses.count(True)
        total_responses = len(correct_responses)

        frac_correct = (num_correct_responses / total_responses)
        print(f"Fraction of correct responses: {frac_correct}")
                

def load_agents(n: int) -> Tuple[nx.Graph, Dict[int, Agent]]:
    """
    Generate a scale free network of n nodes, where each node is an agent.
    
    Args:
    n (int): The number of nodes in the network.
    """

    if n not in [10, 100, 1000]:
        raise ValueError("n must be one of 10, 100, 1000")
    else:
        graph =  nx.read_graphml(f"data/scale_free_network_{n}_nodes.graphml")
        agents_dict = {}
        for id in graph.nodes:
            agents_dict[int(id)] = Agent(id=int(id), name=fake_name(), age=fake_age(), job=fake_job(), hobby=fake_hobby())

        return graph, agents_dict

if __name__ == "__main__":

    test_mmlu()

    pass

