"""
    This script analyse a csv file and make all the required plots.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from typing import Tuple
import pandas as pd
import lib.parse as parse
import lib.analyse as analyse
import lib.visualize as visu

AGENT_RESPONSES_REP = 'output/agent_responses/'
OUTPUT_ANALYSIS_REP = "output/analysis/"

def find_caracteristics(file_path : Path) -> Tuple[int, str, str]:
    '''
        With the file name, determine the number of agents,
    the network bias the network type and if biased nodes are 
    edges or hub nodes.
    '''
    num_agents = int(file_path.name.split('.')[0])

    graph_type = "_".join(file_path.parent.name.split('_')[:-1])

    network_bias = "Unbiased"
    if "incorrect" in str(file_path):
        network_bias = "Incorrect bias"
    elif "correct" in str(file_path):
        network_bias = "Correct bias"

    bias_location = None
    if network_bias != "Unbiased":
        bias_location = file_path.parent.name.split('_')[-1]

    return num_agents, graph_type, network_bias, bias_location

def analyse_simu(agent_response: Path, analyse_dir: Path, figs = False) -> Tuple[Path, str, int, str]:
    '''
        Analyse one network, create the related plot and csv files and return the
    path, the name and the number of agents.
    '''
    # First, we determine the simulation's main characteristics
    num_agents, graph_type, network_bias, bias_location = find_caracteristics(agent_response)

    # Directory creation if they do not exist
    final_res_path = analyse_dir / f'{graph_type}_{bias_location}/'
    final_res_path.mkdir(parents=True, exist_ok=True)

    # Parsing the agent output
    agent_parsed_resp = parse.parse_output_mmlu(agent_response, final_res_path / 'agent_response_parsed.csv')
    network_responses_df = parse.get_network_responses(agent_parsed_resp, final_res_path / 'network_responses.csv')

    ### Analysis one by one :
    # accuracy repartition
    visu.accuracy_repartition(network_responses_df,
                              f'{network_bias} {bias_location}',
                              num_agents,
                              final_res_path)

    # Consensus and simpson consensus
    consensus_df = analyse.calculate_consensus_per_question(agent_parsed_resp)
    visu.consensus_repartition(consensus_df, graph_type, num_agents, final_res_path)

    # Opinion changes
    opinion_changes = analyse.find_evolutions(agent_parsed_resp)
    visu.opinion_changes(opinion_changes, graph_type, num_agents, final_res_path)

    # Figs
    if figs:
        graphml_path = Path(f'experiment/data/{graph_type}/{num_agents}.graphml')
        visu.created_figs(agent_parsed_resp, graphml_path, final_res_path / 'figs/')

    # Wrong response consensus
    agent_parsed_wrong_responses = analyse.filter_wrong_responses(agent_parsed_resp,
                                                             network_responses_df)
    consensus_df = analyse.calculate_consensus_per_question(agent_parsed_wrong_responses)
    visu.consensus_repartition(consensus_df,
                               f'{graph_type} (wrong ansers only)',
                               num_agents,
                               final_res_path,
                               wrong_response= True)

    return final_res_path, graph_type, num_agents, network_bias


def rapid_analyse_simu(agent_response: Path, analyse_dir: Path, figs = False) -> Tuple[Path, str, int, str]:
    # First, we determine what the file is
    graph_type = agent_response.parent.name.split('/')[-1]
    num_agents = int(agent_response.name.split('.')[0])
    network_bias = "Unbiased"
    if "incorrect" in str(agent_response):
        network_bias = "Incorrect bias"
    elif "correct" in str(agent_response):
        network_bias = "Correct bias"
    
    # Directory creation if they do not exist
    final_res_path = analyse_dir / f'{num_agents}_agents/{graph_type}/'
    Path(final_res_path).mkdir(parents=True, exist_ok=True)

    # Parsing the agent output
    agent_parsed_resp = parse.parse_output_mmlu(agent_response, final_res_path / 'agent_response_parsed.csv')

    ### Analysis one by one :
    # Accuracy per round
    network_responses_df = analyse.get_network_responses(agent_parsed_resp)

    

    return final_res_path, graph_type, num_agents, network_bias
