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
    'fully_connected' : 'Fully\nConnected',
    'fully_disconnected' : 'Fully\nDisconnected',
    'random' : 'Random',
    'scale_free_correct_edge' : 'Correct\nBias\n(Edge)',
    'scale_free_correct_hub' : 'Correct\nBias\n(Hub)',
    'scale_free_incorrect_edge' : 'Incorrect\nBias\n(Edge)',
    'scale_free_incorrect_hub' : 'Incorrect\nBias\n(Hub)',
    'scale_free_unbiased' : 'Scale-Free\nUnbiased'
}

GRAPH_COLORS = {
    'fully_connected': '#F7C088',
    'fully_disconnected': '#C1EDED',
    'random': '#C4A2DF',
    'scale_free_correct_edge': '#a0bf74',
    'scale_free_correct_hub': '#a0bf74',
    'scale_free_incorrect_edge': '#d2747a',
    'scale_free_incorrect_hub': '#d2747a',
    'scale_free_unbiased': '#8BA7D4'
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
    
    visu.correct_prop_vs_network(f"{RESULTS_PATH}**/consensus.csv",
                                 RESULTS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.consensus_table(f"{RESULTS_PATH}**/consensus.csv", 
                           RESULTS_PATH)

    visu.neighbours_accuracy(f"{RESULTS_PATH}**/proportion_neighbors_correct_previous_round.csv", 
                            RESULTS_PATH, GRAPH_COLORS)

    analyse.calculate_cost_per_round(f"{RESULTS_PATH}/cost_per_round.csv")

    pass

if __name__ == '__main__':
    main()
