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

AGENT_UNBIASED_RESPONSES_REP = 'output/unbiased/agent_responses/'
OUTPUT = 'output/'
GLOBAL_ANALYSIS = "output/analysis/"

def main():
    Path(GLOBAL_ANALYSIS).mkdir(parents=True, exist_ok=True)

    csv_files = glob(f'{OUTPUT}unbiased/agent_responses/scale_free_network/*.csv', recursive=True)
    csv_files += glob(f'{OUTPUT}correct_biased/agent_responses/scale_free_network/*.csv', recursive=True)
    csv_files += glob(f'{OUTPUT}incorrect_biased/agent_responses/scale_free_network/*.csv', recursive=True)

    # reset result files and write headers
    f = open(f'{GLOBAL_ANALYSIS}network_responses.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['network_bias', 'size', 'question_number', 'round', 'correct'])
    f.close()

    f = open(f'{GLOBAL_ANALYSIS}consensus_comparison.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['network_bias', 'size', 'consensus', 'consensus_simpson', 'simpson_wrong'])
    f.close()

    # Loop on each csv file
    for file in csv_files:
        print(file)

        # We analyse simulations one by one :
        res_dir, _, num_agent, network_bias = network_analysis.analyse_simu(agent_response= Path(file), 
                                                                        analyse_dir= Path(file).parent.parent.parent / "analysis/", 
                                                                        figs = False)
        
        # Then we save results for inter-graph analysis
        # Get the accuracy of group responses
        network_responses = pd.read_csv(res_dir / 'network_responses.csv')   

        # save responses
        network_responses['size'] = num_agent
        network_responses['network_bias'] = f'{network_bias}'
        network_responses[['network_bias', 'size', 
                           'question_number', 
                           'round', 'correct']].to_csv(f'{GLOBAL_ANALYSIS}network_responses.csv',
                                                        mode='a', 
                                                        sep=',', 
                                                        index=False, 
                                                        header=False)

        # Get the consensus for this graph (normal and wrong answers only consensus)
        consensus = pd.read_csv(res_dir / 'consensus.csv')
        consensus = consensus[['consensus', 'simpson']].mean()

        consensus_w = pd.read_csv(res_dir / 'consensus_wrong_response.csv')
        consensus_w = consensus_w[['simpson']].mean()

        # save consensus
        f = open(f'{GLOBAL_ANALYSIS}consensus_comparison.csv', 'a', newline='')
        writer = csv.writer(f)
        writer.writerow([network_bias,
                        num_agent,
                        consensus['consensus'],
                        consensus['simpson'],
                        consensus_w['simpson']])
        f.close()

    ## Graph comparison analyses
    # Accuracy vs round for each size
    network_responses = pd.read_csv(Path(f'{GLOBAL_ANALYSIS}network_responses.csv'))

    accu_vs_round_dir = Path(f'{GLOBAL_ANALYSIS}/accuracy_vs_round/')
    accu_vs_round_dir.mkdir(parents=True, exist_ok= True)

    for size in network_responses['size'].unique():
        response_df_size = network_responses.query(f"size == {size}")

        visu.accurracy_vs_round(response_df_size,
                                int(size),
                                accu_vs_round_dir)
        
    # Accuracy vs graph type and size
    visu.accuracy_vs_agent_number(Path(f'{GLOBAL_ANALYSIS}network_responses.csv'),
                                  Path(f'{GLOBAL_ANALYSIS}'))

    # Consensus vs graph type and size
    visu.consensus_vs_graph(Path(f'{GLOBAL_ANALYSIS}consensus_comparison.csv'),
                            Path(f'{GLOBAL_ANALYSIS}'))

    return

if __name__ == '__main__':
    main()