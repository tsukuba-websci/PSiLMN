"""
    Analyse all responses from all of the agent simulations
"""
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from glob import glob
import lib.visualize as visu
import lib.analyse as analyse
import lib.visualize as visu

AGENT_RESPONSES_REP = 'output/agent_responses/'
OUTPUT_ANALYSIS_REP = "output/analysis/"

OUTPUT_PATH = 'output/'
COMBINED_ANALYSIS_PATH = f"{OUTPUT_PATH}combined/analysis/"

GRAPH_NAMES = {'unbiased' : 'Unbiased',
                'incorrect_bias_hub' : 'Incorrect Bias (Hub)',
                'correct_bias_hub' : 'Correct Bias (Hub)',
                'incorrect_bias_edge' : 'Incorrect Bias (Edge)',
                'correct_bias_edge' : 'Correct Bias (Edge)',
                }

GRAPH_COLORS = {
    'unbiased': '#377eb8',
    'correct_bias_hub': '#4daf4a',
    'correct_bias_edge': '#a6d854',
    'incorrect_bias_hub': '#e41a1c',
    'incorrect_bias_edge': '#f25355',
}

def main():
    Path(COMBINED_ANALYSIS_PATH).mkdir(parents=True, exist_ok=True)

    csv_files = glob(f'{OUTPUT_PATH}**/agent_responses/**/*.csv', recursive=True)

    for file in csv_files:
        analyse.analyse_simu(agent_response= Path(file), 
                                        analyse_dir= Path(file).parent.parent.parent / "analysis/",
                                        graph_names=GRAPH_NAMES,
                                        graph_colors=GRAPH_COLORS,
                                        figs = False)
        

    visu.accuracy_vs_bias(f"{OUTPUT_PATH}**/analysis/**/accuracy_per_network_and_repeat.csv", COMBINED_ANALYSIS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.accuracy_vs_round(f"{OUTPUT_PATH}**/analysis/**/accuracy_per_round.csv", COMBINED_ANALYSIS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.consensus_vs_bias(f"{OUTPUT_PATH}**/analysis/**/consensus.csv", COMBINED_ANALYSIS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    visu.consensus_incorrect_vs_bias(f"{OUTPUT_PATH}**/analysis/**/consensus_wrong_response.csv", COMBINED_ANALYSIS_PATH, GRAPH_NAMES, GRAPH_COLORS)

    return

if __name__ == '__main__':
    main()