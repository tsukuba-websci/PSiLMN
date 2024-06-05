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
from matplotlib.patches import Patch  # Add this import

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

    plt.figure(figsize=(12, 8))
    g = sns.displot(df, x="accuracy", hue="network_number")
    g.set(title=f"Average Accuracy per Question and network for {number_agents} Agents\nin {graph_name}")
    g.set_axis_labels("Accuracy (proportion of correct answers)", "Frequency (%)")
    plt.savefig(res_dir_path / 'accuracy_per_question_and_network.png')

    # accuracy per question and repeat
    df = final_responses.groupby(['question_number', 'repeat'])['correct'].mean().reset_index()
    df = df.rename(columns = {'correct': 'accuracy'})
    df.to_csv(res_dir_path / 'accuracy_per_question_and_repeat.csv', index = False)

    plt.figure(figsize=(12, 8))
    g = sns.displot(df, x="accuracy")
    g.set(title=f"Average Accuracy per Question for {number_agents} Agents\nin {graph_name}")
    g.set_axis_labels("Accuracy (proportion of correct answers)", "Frequency (%)")
    plt.savefig(res_dir_path / 'accuracy_per_question_and_repeat.png')

    # accuracy per network and repeat
    df = final_responses.groupby(['network_number', 'repeat'])['correct'].mean().reset_index()
    df = df.rename(columns = {'correct': 'accuracy'})
    df.to_csv(res_dir_path / 'accuracy_per_network_and_repeat.csv', index = False)

    plt.figure(figsize=(12, 8))
    g = sns.displot(df, x="accuracy")
    g.set(title=f"Average Accuracy per Network for {number_agents} Agents\nin {graph_name}")
    g.set_axis_labels("Accuracy (proportion of correct answers)", "Frequency (%)")
    plt.savefig(res_dir_path / 'accuracy_per_network_and_repeat.png')

    # accuracy per round
    df = network_responses.groupby('round')['correct'].agg(['mean', 'sem']).reset_index()
    df = df.rename(columns={'mean': 'accuracy', 'sem': 'standard_error'})
    df.to_csv(res_dir_path / 'accuracy_per_round.csv', index = False)

    return

def consensus_repartition(consensus_df: pd.DataFrame, res_dir_path: Path, graph_colors: dict[str, str]) -> None:
    ''' Save the combined consensus repartition in res_dir_path location. The program creates separate .png images and .csv files. 
    res_dir_path should lead to a directory, not to a file.
    '''
    # Ensure the directory exists
    Path(res_dir_path).mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    consensus_df.to_csv(res_dir_path / 'consensus.csv', mode='w', sep=',', index=False)

    consensus_correct = consensus_df.query("network_correct == 'true'")
    consensus_incorrect = consensus_df.query("network_correct == 'false'")

    network_type = res_dir_path.name

    # Plot for correct_prop
    plt.figure(figsize=(12, 8))
    sns.histplot(consensus_correct, x="correct_prop", color=graph_colors['scale_free_correct_hub'], stat='probability', alpha=0.6, label='Correct')
    sns.histplot(consensus_incorrect, x="correct_prop", color=graph_colors['scale_free_incorrect_hub'], stat='probability', alpha=0.6, label='Incorrect')
    plt.title("Proportion of Agents Correct per Question", fontsize=24)
    plt.xlabel("Proportion of Agents Correct", fontsize=20)
    plt.ylabel("Relative Frequency", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, 1)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(res_dir_path / 'correct_prop.png', dpi=300)
    plt.close()

    # Plot for simpson
    plt.figure(figsize=(12, 8))
    sns.histplot(consensus_correct, x="simpson", color=graph_colors['scale_free_correct_hub'], stat='probability', alpha=0.5, label='Correct')
    sns.histplot(consensus_incorrect, x="simpson", color=graph_colors['scale_free_incorrect_hub'], stat='probability', alpha=0.5, label='Incorrect')
    plt.title("Consensus within the Collective", fontsize=28)
    plt.xlabel("Simpson Index $\\lambda$", fontsize=28)
    plt.ylabel("Relative Frequency", fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(0.2, 1)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(res_dir_path / f'simpson_{network_type}.png', dpi=300)
    plt.close()

def opinion_changes(df_opinion_evol: pd.DataFrame, bias: str, res_dir_path: Path, graph_names: dict[str, str], graph_colors: dict[str, str]) -> None:
    '''
    Save the opinion change repartition in res_dir_path location.
    The program creates a .png image and a .csv file.
    res_dir_path should lead to a directory, not to a file.
    '''
    graph_colors = {'fully_connected': '#fbe0c3',
    'fully_disconnected': '#e0f6f6',
    'random': '#e2d0ef',
    'scale_free_correct_edge': '#d0dfba',
    'scale_free_correct_hub': '#d0dfba',
    'scale_free_incorrect_edge': '#e8babc',
    'scale_free_incorrect_hub': '#e8babc',
    'scale_free_unbiased': '#c5d3ea'}

    network_type = res_dir_path.name

    # rename evolution column
    df_opinion_evol = df_opinion_evol.rename(columns={'evolution': 'Answer Change'})

    # Correctly format the display names
    title = graph_names.get(bias, bias).replace('\n', ' ').replace('Scale-Free', '')

    # Define the custom palette and the order of the hues
    custom_palette = {
        "I $\\rightarrow$ I": graph_colors['scale_free_incorrect_hub'],
        "C $\\rightarrow$ C": graph_colors['scale_free_correct_hub'],
        "I $\\rightarrow$ C": graph_colors['scale_free_unbiased'],
        "C $\\rightarrow$ I": graph_colors['fully_disconnected']
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

    # Create the plot
    plt.figure(figsize=(12, 8))
    gfg = sns.barplot(data=grouped, x='round', y='percentage', hue='Answer Change', hue_order=hue_order, palette=custom_palette, edgecolor='black')
    plt.xlabel("Round Number", fontsize=20)
    plt.ylabel("Percentage of Agents (%)", fontsize=20)
    plt.ylim(0, 80)
    # Define the custom x-tick labels
    round_transitions = {0: "1 $\\rightarrow$ 2", 1: "2 $\\rightarrow$ 3", 2: "3 $\\rightarrow$ 4"}

    plt.xticks(ticks=range(0, 3), labels=[round_transitions[i] for i in range(0, 3)], fontsize=16)

    plt.yticks(fontsize=16)
    plt.title(title, fontsize=24)
    plt.tight_layout()

    # for legend text
    plt.setp(gfg.get_legend().get_texts(), fontsize='16')
    plt.setp(gfg.get_legend().get_title(), fontsize='20')

    # Save the figure and data
    Path(res_dir_path).mkdir(parents=True, exist_ok=True)
    df_opinion_evol.to_csv(res_dir_path / 'opinion_changes.csv', mode='w', sep=',', index=False)
    plt.savefig(res_dir_path / f'opinion_changes_{network_type}.png', dpi=300)
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

    plt.figure(figsize=(12, 8))
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
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=consensus_df ,x='size', y=Y, hue="network_bias").set(title=f'{title} vs Graph Type and Size')
        plt.xlabel('Number of agents (graph size)')
        plt.ylabel('Consensus (%)')
        plt.grid(True)
        plt.savefig(res_file_path / res_file)
        plt.close('all')

def accuracy_vs_network(input_file_path: str, output_dir: str, human_readable_labels: dict[str, str], graph_colors: dict[str, str]) -> None:
    ''' Plot the accuracy vs bias comparison between graphs. The program creates a .png image and saves it to the output directory. Additionally, it saves the accuracy and standard error for each csv file into a new csv file. '''

    results_df = pd.DataFrame(columns=['network', 'accuracy', 'standard_error'])
    
    # Read all the CSVs
    csv_files = glob.glob(input_file_path, recursive=True)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)['accuracy']
        mean = df.mean()
        standard_error = df.std() / np.sqrt(len(df))

        # Append results
        new_data = pd.DataFrame({
            'network': [Path(csv_file).parent.name],
            'accuracy': mean,
            'standard_error': standard_error
        })

        results_df = pd.concat([results_df if not results_df.empty else None, new_data], ignore_index=True)

    # Save the results to a new CSV file
    results_df.sort_values(by='network', ascending=False, inplace=True)

    results_path = Path(output_dir) / 'accuracy_and_network.csv'
    results_df.to_csv(results_path, index=False)

    # Define categories for unbiased and biased networks
    unbiased_networks = ['scale_free_unbiased', 'random', 'fully_connected', 'fully_disconnected']
    biased_networks = ['scale_free_correct_hub', 'scale_free_incorrect_hub', 'scale_free_correct_edge', 'scale_free_incorrect_edge', 'scale_free_unbiased']

    def plot_accuracy_vs_network(results_df, networks, title, file_name, label_modifier=None):
        filtered_df = results_df[results_df['network'].isin(networks)]
        network_colors = [graph_colors.get(network, 'gray') for network in filtered_df['network']]
        
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(
            filtered_df['network'], 
            filtered_df['accuracy'] * 100, 
            yerr=filtered_df['standard_error'] * 100, 
            capsize=5, 
            color=network_colors, 
            ecolor=(0, 0, 0, 0.3)
        )

        # Apply hatching based on labels
        for bar, network, color in zip(bars, filtered_df['network'], network_colors):
            bar.set_edgecolor("k")  # Set the edge color to match the bar color
            if 'edge' in network:
                bar.set_facecolor((*bar.get_facecolor()[:3], 0.3))
                bar.set_hatch('/')  # diagonal lines
            elif 'hub' in network:
                bar.set_facecolor((*bar.get_facecolor()[:3], 0.3))
                bar.set_hatch('o')  # reverse diagonal lines
            else:
                bar.set_facecolor((*bar.get_facecolor()[:3], 0.5))
                bar.set_hatch('')
        
        plt.xlabel('Network Type', fontsize=20)
        plt.ylabel('Accuracy (%)', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(title, fontsize=24)
        
        # Modify labels if a modifier function is provided
        if label_modifier:
            labels = [label_modifier(human_readable_labels.get(network, network)) for network in filtered_df['network']]
        else:
            labels = [human_readable_labels.get(network, network) for network in filtered_df['network']]
        
        plt.xticks(range(len(filtered_df['network'])), labels)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / file_name, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot for unbiased networks
    plot_accuracy_vs_network(
        results_df,
        unbiased_networks,
        'Accuracy vs Network Type',
        'accuracy_vs_network.png',
        label_modifier=lambda x: 'Scale-Free' if 'Unbiased' in x else x
    )

    # Plot for biased networks
    plot_accuracy_vs_network(
        results_df,
        biased_networks,
        'Accuracy vs Bias Type in Scale-Free Networks',
        'accuracy_vs_bias.png',
        label_modifier=lambda x: 'Unbiased' if 'Scale-Free' in x else x
    )

def accuracy_vs_round(agent_responses_path: str, output_dir: str, human_readable_labels: dict[str, str], graph_colors: dict[str, str]) -> None:
    for results_type in ['structure', 'bias']:
        csv_files = glob.glob(agent_responses_path, recursive=True)
        results_df = pd.DataFrame()
        for csv_file in csv_files:
            if results_type == 'bias' and 'scale_free' in csv_file:
                df = pd.read_csv(csv_file)
                df['network'] = Path(csv_file).parent.name
                results_df = pd.concat([results_df, df], ignore_index=True, sort=False)
            elif results_type == 'structure' and 'edge' not in csv_file and 'hub' not in csv_file:
                df = pd.read_csv(csv_file)
                df['network'] = Path(csv_file).parent.name
                results_df = pd.concat([results_df, df], ignore_index=True, sort=False)

        # Save the combined DataFrame to a CSV file
        combined_csv_path = Path(output_dir) / 'accuracy_and_round.csv'
        results_df.to_csv(combined_csv_path, index=False)

        plt.figure(figsize=(16, 9))  # 16:9 aspect ratio
        sns.set_style("white")

        def format_label(network):
            if 'incorrect' in network:
                return f"Scale-Free (Incorrect Bias {network.split('_')[-1].capitalize()})"
            elif 'correct' in network:
                return f"Scale-Free (Correct Bias {network.split('_')[-1].capitalize()})"
            elif 'unbiased' in network:
                return 'Scale-Free (Unbiased)'
            else:
                return human_readable_labels.get(network, network).replace("\n", " ")

        hatch_patterns = {
            'hub': 'o',
            'edge': '/'
        }

        custom_legend_patches = []

        for network, group in results_df.groupby('network'):
            x = group['round']
            y = group['accuracy'] * 100
            yerr = group['standard_error'] * 100
            custom_color = graph_colors.get(network, 'gray')

            hatch = next((pattern for key, pattern in hatch_patterns.items() if key in network), '')

            plt.plot(x + 1, y, marker='o', markersize=5, label=format_label(network), linewidth=3, color=custom_color)
            plt.fill_between(x + 1, y - yerr, y + yerr, color=custom_color, alpha=0.3, hatch=hatch)

            custom_legend_patches.append(Patch(facecolor=custom_color, edgecolor='k', hatch=hatch, label=format_label(network)))

        plt.xlabel('Round', fontsize=20)
        plt.ylabel('Accuracy (%)', fontsize=20)
        plt.title('Accuracy vs Round', fontsize=24)

        xticks = np.arange(1, len(results_df['round'].unique()) + 1)
        plt.xticks(xticks, fontsize=16)
        plt.yticks(fontsize=16)

        # Set the custom order for the legend items
        custom_legend_order = ['Fully Connected', 'Fully Disconnected', 'Random','Scale-Free (Unbiased)', 'Scale-Free (Correct Bias Hub)', 'Scale-Free (Correct Bias Edge)',
                            'Scale-Free (Incorrect Bias Hub)', 'Scale-Free (Incorrect Bias Edge)']
        custom_legend_patches_sorted = sorted(custom_legend_patches, key=lambda patch: custom_legend_order.index(patch.get_label()))

        plt.legend(handles=custom_legend_patches_sorted, fontsize=14, handlelength=4)
        plt.tight_layout()

        # Save the plot as a PNG file
        plot_path = Path(output_dir) / f'accuracy_vs_round_{results_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def correct_prop_vs_network(input_file_path: str, output_dir: str, human_readable_labels: dict[str, str], graph_colors: dict[str, str]) -> None:
    results_df = pd.DataFrame(columns=['network', "correct_prop", 'standard_error'])
    csv_files = glob.glob(input_file_path, recursive=True)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file).get("correct_prop", pd.Series())
        mean = df.mean()
        sem = df.std() / np.sqrt(len(df))
        results_df = pd.concat([results_df if not results_df.empty else None, pd.DataFrame({'network': [Path(csv_file).parent.name], "correct_prop": mean, 'standard_error': sem})], ignore_index=True)

    results_path = Path(output_dir) / "correct_prop_and_network_type.csv"
    custom_order = ['fully_connected','fully_disconnected', 'random', 'scale_free_unbiased', 'scale_free_incorrect_hub', 'scale_free_incorrect_edge', 'scale_free_correct_hub', 'scale_free_correct_edge']
    results_df['network_order'] = results_df['network'].apply(lambda x: custom_order.index(x))
    results_df = results_df.sort_values(by='network_order')
    results_df.to_csv(results_path, index=False)

    network_colors = [graph_colors.get(network, 'gray') for network in results_df['network']]
    plt.figure(figsize=(12, 8))

    bars = plt.bar(range(len(results_df['network'])), results_df["correct_prop"]*100, yerr=results_df['standard_error']*100, capsize=5, color=network_colors, ecolor=(0, 0, 0, 0.3))

    # Apply hatching based on labels
    for bar, network, color in zip(bars, results_df['network'], network_colors):
        bar.set_edgecolor("k")  # Set the edge color to match the bar color
        if 'edge' in network:
            bar.set_facecolor((*bar.get_facecolor()[:3], 0.3))
            bar.set_hatch('/')  # diagonal lines
        elif 'hub' in network:
            bar.set_facecolor((*bar.get_facecolor()[:3], 0.3))
            bar.set_hatch('o')  # circles
        else:
            bar.set_facecolor((*bar.get_facecolor()[:3], 0.5))
            bar.set_hatch('')

    plt.xlabel('Network Type', fontsize=20)
    plt.ylabel("Percentage of Agents that Answered Correctly (%)", fontsize=20)
    plt.xticks(range(len(results_df['network'])), [human_readable_labels.get(str(network), str(network)) for network in results_df['network']], fontsize=12)
    plt.yticks(fontsize=16)
    plt.ylim(0, 100)
    plt.title("Percentage of Agents that Answered Correctly", fontsize=24)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "correct_prop_vs_network_type.png", dpi=300, bbox_inches='tight')
    plt.close()

def consensus_table(input_file_path: str, output_dir: str):
    """
        Compute average consensus for each graph. Three values are computed for each graph :
    the average consensus, average only for questions the network responded correctly and average
    only for question the network responded wrongly.

    input_file_path is a regular expression leading to all the consensus.csv file of each network.
    res_file_path should lead to a repertory, not to a file.
    """
    csv_files = glob.glob(input_file_path)
    data_res = []

    for file in csv_files:
        df = pd.read_csv(file, delimiter = ',')

        # Network name
        network = file.split("/")[-2]

        # Consensus metrics
        overall = df['simpson'].mean()
        correct = df.query("network_correct == True")['simpson'].mean()
        incorrect = df.query("network_correct == False")['simpson'].mean()

        data_res.append([network, overall, correct, incorrect])
    
    res = pd.DataFrame(columns= ["network_type", 
                                 "overall_consensus", 
                                 "correct_response_consensus", 
                                 "incorrect_response_consensus"],
                        data = data_res)
    res.to_csv(Path(output_dir) / "consensus_table.csv", index = False)
        
def neighbours_accuracy(input_file_path: str, output_file_path: str, graph_colours: dict[str, str]) -> None:
    csv_files = glob.glob(input_file_path, recursive=True)
    csv_files = [file for file in csv_files if "disconnected" not in file]

    if not csv_files:
        return

    df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)
    df = df[['correct_this_round', 'correct_prev_round', 'prop_correct_neighbors']]
    df.to_csv(Path(output_file_path) / 'neighbours_accuracy.csv', index=False)

    # Convert relevant columns to numeric and handle NaN values
    df['correct_this_round'] = pd.to_numeric(df['correct_this_round'], errors='coerce')
    df['correct_prev_round'] = df['correct_prev_round'].astype(bool)
    df['prop_correct_neighbors'] = pd.to_numeric(df['prop_correct_neighbors'], errors='coerce')
    df = df.dropna()

    # Create the lmplot with Seaborn
    custom_palette = {True: graph_colours['scale_free_correct_hub'], False: graph_colours['scale_free_incorrect_hub']}
    lm = sns.lmplot(
        data=df,
        x='prop_correct_neighbors',
        y='correct_this_round',
        hue='correct_prev_round',
        palette=custom_palette,
        height=8,
        aspect=1.5,
        legend=False,
        x_estimator=np.mean,
        scatter_kws={'alpha': 0.3, 'edgecolor': 'black', 'linewidth': 0.5}  # Set the opacity of error bars
    )

    for ax in lm.axes.ravel():
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add the title using plt
    plt.title('Agent Influence', fontsize=24)

    # Add axis labels using plt
    plt.xlabel(r'Proportion of Neighbours Correct in Round $n-1$', fontsize=20)
    plt.ylabel(r'Probability of Individual Correct in Round $n$', fontsize=20)

    # Customize ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Create custom legend
    correct_patch = Patch(color=graph_colours['scale_free_correct_hub'], label='Correct', alpha=1)
    incorrect_patch = Patch(color=graph_colours['scale_free_incorrect_hub'], label='Incorrect', alpha=1)
    plt.legend(handles=[correct_patch, incorrect_patch], title=r'Individuals Answer Round $n-1$', fontsize=16, title_fontsize=16, loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 1])

    # Save plot
    plt.savefig(Path(output_file_path) / 'neighbours_accuracy.png', dpi=300)

def created_gifs(parsed_agent_response: pd.DataFrame,
                 graphml_path: Path,
                 res_repertory: Path,
                 network_num: int,
                 graph_colors: dict[str, str]) -> None:
    '''
        Fill res_repertory with an animated Gif for each question of the simulation.
    '''
    df = parsed_agent_response

    if '/correct_bias' in str(res_repertory):
        bias = 'correct_bias'
    elif '/incorrect_bias' in str(res_repertory):
        bias = 'incorrect_bias'

    num_agents = df['agent_id'].unique().max() + 1

    G = nx.read_graphml(graphml_path)
    pos = nx.spring_layout(G, seed=42)

    rounds = df['round'].unique()
    questions = df['question_number'].unique()

    for repeat in range(3):

        res_repertory_repeat = res_repertory / f'repeat_{repeat}'
        res_repertory_repeat.mkdir(parents=True, exist_ok=True)

        for question in questions:
            images = []

            for round_ in rounds:
                round_df = df[(df['round'] == round_) & (df['question_number'] == question) & (df['repeat'] == repeat) & (df['network_number'] == network_num)]
                agent_ids = round_df['agent_id'].unique()
                missing_agents = list(set(range(num_agents)) - set(agent_ids))

                color_map = ['blue'] * num_agents  # Initialize all agents to blue

                # Coloring for missing agents based on bias
                for agent_id in missing_agents:
                    if bias == 'correct_bias':
                        color_map[agent_id] = graph_colors['correct_bias_hub']
                    else:
                        color_map[agent_id] = graph_colors['incorrect_bias_hub']

                # Coloring for agents present in the round
                for agent_id in range(num_agents):
                    if agent_id in round_df['agent_id'].values:
                        row = round_df[round_df['agent_id'] == agent_id].iloc[0]
                        color_map[agent_id] = graph_colors['correct_bias_edge'] if row['correct'] else graph_colors['incorrect_bias_edge']

                plt.figure(figsize=(8, 8))

                pos = nx.spring_layout(G, seed=42)  # Position nodes using the spring layout
                nx.draw_networkx_nodes(G, pos, node_size=500, node_color=color_map, alpha=1)
                nx.draw_networkx_edges(G, pos, alpha=1)
                plt.axis('off')

                # Add round number text above the plot
                round_text = f'Round: {round_ + 1}'
                text_x = 0.05
                text_y = 0.95
                text_color = '#333333'
                text_font = 'Arial'
                text_size = 16
                text_weight = 'bold'

                plt.text(text_x, text_y, round_text, transform=plt.gca().transAxes, fontsize=text_size, 
                        fontname=text_font, fontweight=text_weight, color=text_color, 
                        verticalalignment='top')

                image_path = res_repertory_repeat / f'{num_agents}_q{question}_r{round_+1}.png'
                plt.savefig(image_path, dpi=300)
                plt.close('all')

                images.append(image_path)

            gif_path = res_repertory_repeat / f'q{question}.gif'
            with imageio.get_writer(gif_path, mode='I', fps=1, loop=0) as writer:
                for image_path in images:
                    image = imageio.v3.imread(image_path)
                    writer.append_data(image)

            for image_path in images:
                os.remove(image_path)
