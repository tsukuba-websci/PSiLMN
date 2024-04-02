"""
    Contains drawing functions. This functions turn a dataFrame into
a graph that is saved in the destination path. Each graph is saved with
the related data in a csv file.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import seaborn as sns
import glob
import os
import numpy as np

from pathlib import Path
import glob
import numpy as np
from pathlib import Path

### Single simulation plot
def accuracy_repartition(network_responses : pd.DataFrame, 
                         graph_name : str, 
                         number_agents : int, 
                         res_dir_path : Path) -> None:
    '''
        Plot the frequency of accuracy for each question on each
    graph.
    '''
    Path(res_dir_path).mkdir(parents=True, exist_ok=True)
    last_round = network_responses['round'].unique().max()
    final_responses = network_responses.query(f"round == {last_round}")

    # accuracy per question and per network
    df = final_responses.groupby(['network_number', 'question_number'])['correct'].mean().reset_index()
    df = df.rename(columns = {'correct': 'accuracy'})
    df.to_csv(res_dir_path / 'accuracy_per_question_and_network.csv', index = False)

    plt.figure(figsize=(16, 9))
    g = sns.displot(df, x="accuracy", hue="network_number")
    g.set(title=f"Average Accuracy per Question and network for {number_agents} Agents\nin {graph_name}")
    g.set_axis_labels("Accuracy (proportion of correct answers)", "Frequency (%)")
    plt.savefig(res_dir_path / 'accuracy_per_question_and_network.png')

    # accuracy per question and repeat
    df = final_responses.groupby(['question_number', 'repeat'])['correct'].mean().reset_index()
    df = df.rename(columns = {'correct': 'accuracy'})
    df.to_csv(res_dir_path / 'accuracy_per_question_and_repeat.csv', index = False)

    plt.figure(figsize=(16, 9))
    g = sns.displot(df, x="accuracy")
    g.set(title=f"Average Accuracy per Question for {number_agents} Agents\nin {graph_name}")
    g.set_axis_labels("Accuracy (proportion of correct answers)", "Frequency (%)")
    plt.savefig(res_dir_path / 'accuracy_per_question_and_repeat.png')

    # accuracy per network and repeat
    df = final_responses.groupby(['network_number', 'repeat'])['correct'].mean().reset_index()
    df = df.rename(columns = {'correct': 'accuracy'})
    df.to_csv(res_dir_path / 'accuracy_per_network_and_repeat.csv', index = False)

    plt.figure(figsize=(16, 9))
    g = sns.displot(df, x="accuracy")
    g.set(title=f"Average Accuracy per Network for {number_agents} Agents\nin {graph_name}")
    g.set_axis_labels("Accuracy (proportion of correct answers)", "Frequency (%)")
    plt.savefig(res_dir_path / 'accuracy_per_network_and_repeat.png')

    # accuracy per round
    df = network_responses.groupby('round')['correct'].agg(['mean', 'sem']).reset_index()
    df = df.rename(columns={'mean': 'accuracy', 'sem': 'standard_error'})
    df.to_csv(res_dir_path / 'accuracy_per_round.csv', index = False)

    return

def consensus_repartition(consensus_df : pd.DataFrame,
                            graph_name : str,
                            number_agents : int,
                            res_dir_path : Path,
                            wrong_response = False) -> None:
    '''
        Save the consensus repartition in res_dir_path location. The
    programm create a .png image and a .csv file.
    res_dir_path should lead to a repertory, not to a file.
    '''
    # csv
    Path(res_dir_path).mkdir(parents=True, exist_ok=True)
    if not wrong_response:
        consensus_df.to_csv(res_dir_path / f'consensus.csv',
                            mode = 'w',
                            sep = ',',
                            index=False)
    else:
        consensus_df.to_csv(res_dir_path / f'consensus_wrong_response.csv',
                            mode = 'w',
                            sep = ',',
                            index=False)
                
    # consensus
    g = sns.displot(consensus_df, x="correct_prop")
    # g.set_theme(rc={'figure.figsize':(11.7,8.27)})
    g.set(title=f"Average Consensus per Question for {number_agents} Agents\nin {graph_name}")
    g.set_axis_labels("Consensus (proportion of correct answers)", "Frequency (%)")
    if not wrong_response:
        plt.savefig(res_dir_path / f'consensus.png')
    else:
        plt.savefig(res_dir_path / f'consensus_wrong_response.png')

    # simpson
    plt.figure(figsize=(16, 9))
    sns.displot(consensus_df, x="simpson")
    plt.title(f"Average Simpson Consensus per Question for {number_agents} Agents\nin {graph_name}")
    plt.xlabel("Simpson probability")
    plt.ylabel("Frequency (%)")

    if not wrong_response:
        plt.savefig(res_dir_path / f'simpson.png')
    else:
        plt.savefig(res_dir_path / f'simpson_wrong_responses.png')
    plt.close('all')

def opinion_changes(df_opinion_evol: pd.DataFrame,
                    graph_name: str,
                    res_dir_path: Path) -> None:
    '''
    Save the opinion change repartition in res_dir_path location.
    The program creates a .png image and a .csv file.
    res_dir_path should lead to a directory, not to a file.
    '''

    # rename evolution column
    df_opinion_evol = df_opinion_evol.rename(columns = {'evolution': 'Answer Change'})

    print(df_opinion_evol)

    # Correctly format the display names
    graph_name = graph_name.replace("_", " ").title()

    # Define the custom palette and the order of the hues
    custom_palette = {
        "I $\\rightarrow$ I": "#e41a1c",  # Cornflower Blue
        "C $\\rightarrow$ C": "#4daf4a",  # Golden Rod
        "I $\\rightarrow$ C": "#a6d854",  # Lime Green
        "C $\\rightarrow$ I": "#f0898A"   # Dark Orange
    }
    hue_order = ["C $\\rightarrow$ C", "I $\\rightarrow$ I", "I $\\rightarrow$ C", "C $\\rightarrow$ I"]

    df_opinion_evol['Answer Change'] = df_opinion_evol['Answer Change'].replace({
        'C -> C': 'C $\\rightarrow$ C',
        'I -> C': 'I $\\rightarrow$ C',
        'C -> I': 'C $\\rightarrow$ I',
        'I -> I': 'I $\\rightarrow$ I'
    })
    grouped = df_opinion_evol.groupby(['round', 'Answer Change']).size().reset_index(name='counts')

    # Step 2: Calculate the percentage for each 'Answer Change' within each 'round'
    grouped['total_per_round'] = grouped.groupby('round')['counts'].transform('sum')  # Sum per 'round'
    grouped['percentage'] = (grouped['counts'] / grouped['total_per_round']) * 100  # Calculate percentage

    print(df_opinion_evol['round'].unique())

    # Create the plot
    plt.figure(figsize=(10, 6))
    gfg = sns.barplot(data=grouped, x='round', y='percentage', hue='Answer Change', hue_order=hue_order, palette=custom_palette)

    plt.xlabel("Round Number", fontsize=20)
    plt.ylabel("Percentage of Agents (%)", fontsize=20)
    round_transitions = {1: "1 $\\rightarrow$ 2", 2: "2 $\\rightarrow$ 3", 3: "3 $\\rightarrow$ 4", 4: "4"}
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Answer Changes', fontsize=24)
    plt.tight_layout()

    # for legend text
    plt.setp(gfg.get_legend().get_texts(), fontsize='16')
    plt.setp(gfg.get_legend().get_title(), fontsize='20')

    # Save the figure and data
    Path(res_dir_path).mkdir(parents=True, exist_ok=True)
    df_opinion_evol.to_csv(res_dir_path / 'opinion_changes.csv', mode='w', sep=',', index=False)
    plt.savefig(res_dir_path / 'opinion_changes.png', dpi=300)
    plt.close('all')

### Cross simulation plot
# This programms take a csv file rather than a panda dataFrame
def accuracy_vs_agent_number(network_responses_path: Path,
                            res_dir_path: Path) -> None:
    '''
        Plot the accuracy comparison between graphs.
    The programm create a .png image.
    res_dir_path should lead to a repertory, not to a file.
    '''
    network_responses_df = pd.read_csv(network_responses_path)
    network_responses_df = network_responses_df.query('round == 2')
    
    err_palette = {"color": "black"}

    plt.figure(figsize=(16, 9))
    sns.pointplot(data = network_responses_df, 
                 x='size', y='correct',
                 errorbar=("ci", 95),
                 err_kws=err_palette,
                 hue='network_bias').set(title="Accuracy vs Graph Size and Graph Type")
    plt.xlabel('Number of agents')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    Path(res_dir_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(res_dir_path / 'accuracy_vs_number_of_agents.png')
    plt.close('all')


def consensus_vs_graph(consensus_path: Path,
                        res_file_path: Path) -> None:
    '''
        Plot the consensus comparisons between graphs and save them
    into res_file_path.
    The comparison metrics included are consensus, simpson and simpson
    on wrong responses.
    res_file_path should lead to a repertory, not to a file.
    '''
    consensus_df = pd.read_csv(consensus_path, sep = ',')
    Y_list = ['consensus','consensus_simpson','simpson_wrong']
    title_list = ['Agents good answers rate',
                'Simpson Consensus',
                'Simpson Consensus For wrong Answers']
    files_name = ['consensus_comparison.png',
                  'simpson_comparison.png',
                  'simpson_wrong_answers.png']
    for Y, title, res_file in zip(Y_list, title_list, files_name):
        plt.figure(figsize=(16, 9))
        sns.lineplot(data=consensus_df ,x='size', y=Y, hue="network_bias").set(title=f'{title} vs Graph Type and Size')
        plt.xlabel('Number of agents (graph size)')
        plt.ylabel('Consensus (%)')
        plt.grid(True)
        plt.savefig(res_file_path / res_file)
        plt.close('all')

def accuracy_vs_bias(input_file_path: str, output_dir: str, human_readable_labels: dict[str, str], graph_colors: dict[str, str]) -> None:
    ''' Plot the accuracy vs bias comparison between graphs. The program creates a .png image and saves it to the output directory. Additionally, now it saves the accuracy and standard error for each csv file into a new csv file. '''

    results_df = pd.DataFrame(columns=['network', 'accuracy', 'standard_error'])
    
    # Read all the CSVs
    csv_files = glob.glob(input_file_path, recursive=True)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)['accuracy']
        mean = df.mean()
        sem = df.std() / np.sqrt(len(df))

        # Append results
        results_df = pd.concat([results_df, pd.DataFrame({'network': [Path(csv_file).parent.name],
                                                          'accuracy': mean,
                                                          'standard_error': sem})], ignore_index=True)

    # Save the results to a new CSV file
    results_df.sort_values(by='accuracy', ascending=False, inplace=True)

    results_path = Path(output_dir) / 'accuracy_and_bias.csv'
    results_df.to_csv(results_path, index=False)

    network_colors = [graph_colors.get(network, 'gray') for network in results_df['network']]

    plt.figure(figsize=(12, 8))
    plt.bar(results_df['network'], results_df['accuracy'], yerr=results_df['standard_error'], capsize=5, color=network_colors)
    plt.xlabel('Network Type', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Accuracy vs Bias Type', fontsize=24)

    # Set the x-tick labels to the human-readable labels
    plt.xticks(range(len(results_df['network'])), [human_readable_labels.get(network, network) for network in results_df['network']], rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'accuracy_vs_bias.png', dpi=300,bbox_inches='tight')

def accuracy_vs_round(agent_responses_path: str, output_dir: str, human_readable_labels: dict[str, str], graph_colors: dict[str, str]) -> None:

    csv_files = glob.glob(agent_responses_path, recursive=True)
    results_df = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['network'] = Path(csv_file).parent.name
        results_df = pd.concat([results_df, df], ignore_index=True, sort=False)

    # Save the combined DataFrame to a CSV file
    combined_csv_path = Path(output_dir) / 'accuracy_and_round.csv'
    results_df.to_csv(combined_csv_path, index=False)

    plt.figure(figsize=(12, 8))
    sns.set_style("white")

    for network, group in results_df.groupby('network'):
        x = group['round']
        y = group['accuracy']
        custom_color = graph_colors.get(network, 'gray')
        plt.plot(x+1, y, marker='o', markersize=5, label=network, linewidth=3, color=custom_color)  # Increase markersize and linewidth
        plt.errorbar(x+1, y, yerr=group['standard_error'], fmt='none', capsize=5, elinewidth=2, ecolor='black')  # Increase capsize and elinewidth

    plt.xlabel('Round', fontsize=20)  # Increase fontsize
    plt.ylabel('Accuracy (%)', fontsize=20)  # Increase fontsize

    plt.title('Accuracy vs Round', fontsize=24)  # Increase fontsize

    xticks = np.arange(1, len(results_df['round'].unique()) + 1)
    plt.xticks(xticks, fontsize=16)
    plt.yticks(fontsize=16)

    # Use the label_mapping for the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [human_readable_labels[label] for label in labels]
    plt.legend(handles, new_labels, fontsize=14)  # Increase fontsize
    plt.tight_layout()

    # Save the plot as a PNG file
    plot_path = Path(output_dir) / 'accuracy_vs_round.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

def consensus_vs_bias(input_file_path: str, output_dir: str, human_readable_labels: dict[str, str], graph_colors: dict[str, str]) -> None:

    consensus_types = {'correct_prop': 'Consensus to Correct Answer', 'simpson': 'Simpson Consensus'}

    for consensus_type, consensus_label in consensus_types.items():

        results_df = pd.DataFrame(columns=['network', consensus_type, 'standard_error'])
        csv_files = glob.glob(input_file_path, recursive=True)

        print(csv_files)

        for csv_file in csv_files:
            df = pd.read_csv(csv_file).get(consensus_type, pd.Series())
            mean = df.mean()
            sem = df.std() / np.sqrt(len(df))
            results_df = pd.concat([results_df, pd.DataFrame({'network': [Path(csv_file).parent.name], consensus_type: mean, 'standard_error': sem})], ignore_index=True)

        results_path = Path(output_dir) / f'{consensus_type}_and_bias.csv'
        results_df.to_csv(results_path, index=False)

        network_colors = [graph_colors.get(network, 'gray') for network in results_df['network']]
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(results_df['network'])), results_df[consensus_type], yerr=results_df['standard_error'], capsize=5, color=network_colors)
        plt.xlabel('Network Type', fontsize=20)
        plt.ylabel(f'{consensus_label}', fontsize=20)
        plt.xticks(range(len(results_df['network'])), [human_readable_labels.get(str(network), str(network)) for network in results_df['network']], rotation=45, ha="right", fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(f'{consensus_label} vs Bias Type', fontsize=24)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'{consensus_type}_vs_bias.png', dpi=300, bbox_inches='tight')

### Gif functions :
def created_figs(parsed_agent_response: pd.DataFrame,
                 graphml_path: Path,
                 res_repertory: Path) -> None:
    '''
        Fill res_repertory with an animated Gif for each question of the
    simulation.
    '''
    df = parsed_agent_response

    num_agents = df['agent_id'].unique().max() + 1

    graph = nx.read_graphml(graphml_path)
    pos = nx.spring_layout(graph)

    rounds = df['round'].unique()
    questions = df['question_number'].unique()

    for question in questions:
        images = []  # To store paths of images for the current question
        for round_ in rounds:
            round_df = df[(df['round'] == round_) & (df['question_number'] == question)]
            color_map = ['green' if row['correct'] else 'red' for _, row in round_df.iterrows()]

            plt.figure(figsize=(10, 8))
            nx.draw(graph, pos=pos, node_color=color_map, with_labels=True, node_size=700)
            plt.title(f'Q{question} R{round_}')

            # Add round number text on the plot
            plt.text(0.05, 0.95, f'Round: {round_}', 
                     transform=plt.gca().transAxes, 
                     fontsize=14, verticalalignment='top', 
                     bbox=dict(boxstyle="round", 
                               alpha=0.5, 
                               facecolor='white'))

            # Save the plot for the current round
            image_path = res_repertory / f'/{num_agents}_q{question}_r{round_+1}.png'

            # Ensure the directory exists
            os.makedirs(res_repertory, exist_ok=True)
            plt.savefig(image_path)
            plt.close('all')

            images.append(image_path)

        # Create a GIF for the current question
        gif_path = res_repertory / f'{num_agents}_q{question}.gif'
        with imageio.get_writer(gif_path, mode='I', fps=1, loop=0) as writer:
            for image_path in images:
                image = imageio.v3.imread(image_path)
                writer.append_data(image)
        
        # Optional: Remove the individual round images to clean up
        for image_path in images:
            os.remove(image_path)
