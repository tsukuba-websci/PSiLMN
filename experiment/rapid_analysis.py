import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pandas as pd

from pathlib import Path
from glob import glob

import lib.parse as parse
import lib.analyse as analyse
import lib.visualize as visu
import network_analysis

root_dir = Path(__file__).parent.parent

GLOBAL_ANALYSIS = "output/analysis/"
GRAPH = 'experiment/data/'

GRAPH_NAMES = {'unbiased' : 'Unbiased',
                'incorrect_bias_hub' : 'Incorrect Bias (Hub)',
                'correct_bias_hub' : 'Correct Bias (Hub)',
                'incorrect_bias_edge' : 'Incorrect Bias (Edge)',
                'correct_bias_edge' : 'Correct Bias (Edge)',
                }

GRAPH_COLORS = {
    'unbiased': '#377eb8',  # Vivid blue
    'correct_bias_hub': '#4daf4a',  # Vivid green
    'correct_bias_edge': '#a6d854',  # Bright lime green
    'incorrect_bias_hub': '#e41a1c',  # Bright red
    'incorrect_bias_edge': '#f768a1',  # pink
}

def main():
    csv_files = glob('output/**/agent_responses/**/*.csv', recursive=True)

    # reset result files and write headers
    Path(GLOBAL_ANALYSIS).mkdir(parents=True, exist_ok=True)

    # for agent_responses_str in csv_files:
    #     agent_responses_path = Path(agent_responses_str)

    #     graph_type = agent_responses_path.parent.name.split('/')[-1]
    #     num_agents = int(agent_responses_path.name.split('.')[0])
    #     print(graph_type, num_agents)

    #     network_bias = "unbiased"
    #     if ("incorrect" in str(agent_responses_path)) and ("hub" in str(agent_responses_path)):
    #         network_bias = "incorrect_bias_hub"
    #     elif "incorrect" in str(agent_responses_path) and "edge" in str(agent_responses_path):
    #         network_bias = "incorrect_bias_edge"
    #     elif "correct" in str(agent_responses_path) and "hub" in str(agent_responses_path):
    #         network_bias = "correct_bias_hub"
    #     elif "correct" in str(agent_responses_path) and "edge" in str(agent_responses_path):
    #         network_bias = "correct_bias_edge"

    #     res_dir = Path(GLOBAL_ANALYSIS) / network_bias
    #     res_dir.mkdir(parents=True, exist_ok=True)

    #     agent_parsed_resp = parse.parse_output_mmlu(agent_responses_path, res_dir / 'agent_response_parsed.csv')

    #     visu.accuracy_repartition(agent_parsed_resp,
    #                             f'{GRAPH_NAMES[network_bias]}',
    #                             num_agents,
    #                             res_dir)
        
    visu.accuracy_vs_bias(f"{GLOBAL_ANALYSIS}**/accuracy_per_network_and_repeat.csv", f"{GLOBAL_ANALYSIS}/combined/", GRAPH_NAMES, GRAPH_COLORS)

    visu.accuracy_vs_round_ciaran(f"{GLOBAL_ANALYSIS}**/accuracy_per_round.csv", f"{GLOBAL_ANALYSIS}/combined/", GRAPH_NAMES, GRAPH_COLORS)



if __name__ == "__main__":
    main()