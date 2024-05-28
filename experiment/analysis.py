"""
    Analyse all responses from all of the agent simulations
"""
from pathlib import Path

root_dir = Path(__file__).parent

from glob import glob
import lib.visualize as visu
import lib.analyse as analyse
import lib.visualize as visu

AGENT_RESPONSES_REP = f'{root_dir}/output/'
OUTPUT_ANALYSIS_REP = f'{root_dir}/analysis/'

PARSED_RESPONSES_PATH = f"{OUTPUT_ANALYSIS_REP}parsed_responses/"
NETWORK_REPONSES_PATH = f"{OUTPUT_ANALYSIS_REP}network_responses/"
FIGS_PATH = f"{OUTPUT_ANALYSIS_REP}figs/"

GRAPH_NAMES = {
    'fully_connected' : 'Fully Connected',
    'fully_disconnected' : 'Fully Disconnected',
    'random' : 'Random',
    'scale_free_correct_edge' : 'Correct Bias (Edge)',
    'scale_free_correct_hub' : 'Correct Bias (Hub)',
    'scale_free_incorrect_edge' : 'Incorrect Bias (Edge)',
    'scale_free_incorrect_hub' : 'Incorrect Bias (Hub)',
    'scale_free_unbiased' : 'Scale Free Unbiased'
}

GRAPH_COLORS = {
    'fully_connected' : '#a6d854',
    'fully_disconnected' : '#4daf4a',
    'random' : '#f25355',
    'scale_free_correct_edge': '#a6d854',
    'scale_free_correct_hub': '#4daf4a',
    'scale_free_incorrect_edge': '#f25355',
    'scale_free_incorrect_hub': '#e41a1c',
    'scale_free_unbiased': '#377eb8'
}

def main():
    Path(PARSED_RESPONSES_PATH).mkdir(parents=True, exist_ok=True)
    Path(NETWORK_REPONSES_PATH).mkdir(parents=True, exist_ok=True)
    Path(FIGS_PATH).mkdir(parents=True, exist_ok=True)

    response_dirs = [Path(str_path) for str_path in glob(f'{AGENT_RESPONSES_REP}*', recursive=False)]
    
    for response_path in response_dirs:
        print(response_path.name)
        analyse.analyse_simu(agent_response= response_path, 
                            analyse_dir= Path(OUTPUT_ANALYSIS_REP),
                            graph_names=GRAPH_NAMES,
                            graph_colors=GRAPH_COLORS,
                            gifs = False)

    visu.accuracy_vs_bias(f"{FIGS_PATH}scale_free_**/accuracy_per_network_and_repeat.csv", 
                          FIGS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.accuracy_vs_round(f"{FIGS_PATH}**/accuracy_per_round.csv", 
                           FIGS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.consensus_vs_bias(f"{FIGS_PATH}scale_free_**/consensus.csv", 
                           FIGS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.consensus_incorrect_vs_bias(f"{FIGS_PATH}scale_free_**/consensus_wrong_response.csv", 
                                     FIGS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    # visu.neighbours_accuracy(f"{FIGS_PATH}**/proportion_neighbors_correct.csv", 
    #                          FIGS_PATH, GRAPH_COLORS)

    return

if __name__ == '__main__':
    main()