"""
    Analyse all the simulations output in agent_response
"""
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from glob import glob
import csv
import pandas as pd

import network_analysis
import lib.parse as parse
import lib.analyse as analyse
import lib.visualize as visu

AGENT_RESPONSES_REP = 'output/agent_responses/'
OUTPUT_ANALYSIS_REP = "output/analysis/"

GRAPH_PRETY_NAMES = {'fully_connected_network' : 'Fully Connected Network',
                     'fully_disconnected_network' : 'Fully Disconnected Network',
                     'scale_free_network' : 'Scale Free Network',
                     'random_network' : 'Random Network',
                     'watts_strogatz_network' : 'Watts-Strogatz Network'}

def main():
    csv_files = glob(f'{AGENT_RESPONSES_REP}**/*.csv', recursive=True)

    Path(OUTPUT_ANALYSIS_REP).mkdir(parents=True, exist_ok=True)
    # TODO : reset file if it does not already exists

    # reset result files and write headers
    f = open(f'{OUTPUT_ANALYSIS_REP}accuracy.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['round', 'accuracy', 'size', 'graph_type'])
    f.close()

    f = open(f'{OUTPUT_ANALYSIS_REP}consensus_comparison.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['graph_type', 'size', 'consensus', 'consensus_simpson', 'simpson_wrong'])
    f.close()

    # Loop on each csv file
    # for file in csv_files:
    #     print(file)
    #     # We analyse simulations one by one :
    #     res_dir, graph_type, num_agent = network_analysis.analyse_simu(agent_response= Path(file), 
    #                                                                     analyse_dir= Path(OUTPUT_ANALYSIS_REP), 
    #                                                                     figs = False)
        
    #     # Then we save results for inter-graph analysis
    #     # Get the accuracy of group responses
    #     accuracy = pd.read_csv(res_dir / 'accuracy_per_round.csv')   

    #     # save accuracy
    #     accuracy['size'] = num_agent
    #     accuracy['graph_type'] = f'{GRAPH_PRETY_NAMES[graph_type]}'
    #     accuracy[['round', 'accuracy', 'size', 'graph_type']].to_csv(f'{OUTPUT_ANALYSIS_REP}accuracy.csv', 
    #                                                                 mode='a', 
    #                                                                 sep=',', 
    #                                                                 index=False, 
    #                                                                 header=False)

    #     # Get the consensus for this graph (normal and wrong answers only consensus)
    #     consensus = pd.read_csv(res_dir / 'consensus.csv')
    #     consensus = consensus[['consensus', 'simpson']].mean()

    #     consensus_w = pd.read_csv(res_dir / 'consensus_wrong_response.csv')
    #     consensus_w = consensus_w[['simpson']].mean()

    #     # save consensus
    #     f = open(f'{OUTPUT_ANALYSIS_REP}consensus_comparison.csv', 'a', newline='')
    #     writer = csv.writer(f)
    #     writer.writerow([GRAPH_PRETY_NAMES[graph_type],
    #                         num_agent,
    #                         consensus['consensus'],
    #                         consensus['simpson'],
    #                         consensus_w['simpson']])
    #     f.close()

    ## Graph comparison analyses
    # Accuracy vs round for each size
    accuracy_df = pd.read_csv(Path(f'{OUTPUT_ANALYSIS_REP}accuracy.csv'))

    for size in accuracy_df['size'].unique():
        size_directory = Path(f'{OUTPUT_ANALYSIS_REP}{size}_agents/')

        accuracy_df_size = accuracy_df.query(f"size == {size}")

        visu.accurracy_vs_round(accuracy_df_size,
                                int(size),
                                size_directory)
        
    # Accuracy vs graph type and size
    visu.accuracy_vs_agent_number(Path(f'{OUTPUT_ANALYSIS_REP}accuracy.csv'),
                                  Path(f'{OUTPUT_ANALYSIS_REP}'))

    # Consensus vs graph type and size
    visu.consensus_vs_graph(Path(f'{OUTPUT_ANALYSIS_REP}consensus_comparison.csv'),
                            Path(f'{OUTPUT_ANALYSIS_REP}'))

    return

if __name__ == '__main__':
    main()