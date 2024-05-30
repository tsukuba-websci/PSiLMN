"""
    Parse, analyse and create figures from the agent responses.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from glob import glob
import lib.visualize as visu
import lib.analyse as analyse
import lib.visualize as visu

AGENT_RESPONSES_PATH = f'{root_dir}/experiment/output/agent_responses/'
OUTPUT_ANALYSIS_PATH = f'{root_dir}/experiment/output/analysis/'
PARSED_RESPONSES_PATH = f"{OUTPUT_ANALYSIS_PATH}parsed_responses/"
NETWORK_REPONSES_PATH = f"{OUTPUT_ANALYSIS_PATH}network_responses/"
RESULTS_PATH = f"{OUTPUT_ANALYSIS_PATH}results/"

GRAPH_NAMES = {
    'fully_connected' : 'Fully Connected',
    'fully_disconnected' : 'Fully Disconnected',
    'random' : 'Random',
    'scale_free_correct_edge' : 'Correct Bias (Edge)',
    'scale_free_correct_hub' : 'Correct Bias (Hub)',
    'scale_free_incorrect_edge' : 'Incorrect Bias (Edge)',
    'scale_free_incorrect_hub' : 'Incorrect Bias (Hub)',
    'scale_free_unbiased' : 'Scale-Free Unbiased'
}

GRAPH_COLORS = {
    'fully_connected': '#ffd4a0',
    'fully_disconnected': '#ffe0b5',
    'random': '#80c6cb',
    'scale_free_correct_edge': '#d2e1a9',
    'scale_free_correct_hub': '#bbd785',
    'scale_free_incorrect_edge': '#ffbcc5',
    'scale_free_incorrect_hub': '#f98b98',
    'scale_free_unbiased': '#bfe9e9'
}

def main():
    Path(PARSED_RESPONSES_PATH).mkdir(parents=True, exist_ok=True)
    Path(NETWORK_REPONSES_PATH).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)

    response_dirs = [Path(str_path) for str_path in glob(f'{AGENT_RESPONSES_PATH}*', recursive=False)]
    for response_path in response_dirs:
        analyse.analyse_simu(agent_response=response_path, 
                            analyse_dir=Path(OUTPUT_ANALYSIS_PATH),
                            graph_names=GRAPH_NAMES,
                            graph_colors=GRAPH_COLORS,
                            gifs = False)
        visu.neighbours_accuracy(f"{RESULTS_PATH}/{response_path.name}/**/proportion_neighbors_correct_previous_round.csv", 
                        f"{RESULTS_PATH}/{response_path.name}/", GRAPH_COLORS)

    visu.accuracy_vs_network(f"{RESULTS_PATH}**/accuracy_per_network_and_repeat.csv", 
                          RESULTS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.accuracy_vs_round(f"{RESULTS_PATH}**/accuracy_per_round.csv", 
                           RESULTS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.consensus_vs_bias(f"{RESULTS_PATH}scale_free_**/consensus.csv", 
                           RESULTS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.consensus_incorrect_vs_bias(f"{RESULTS_PATH}scale_free_**/consensus_wrong_response.csv", 
                                     RESULTS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.neighbours_accuracy(f"{RESULTS_PATH}**/proportion_neighbors_correct_previous_round.csv", 
                            RESULTS_PATH, GRAPH_COLORS)

    pass

if __name__ == '__main__':
    main()
