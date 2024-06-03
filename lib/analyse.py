"""
    Contains programms to analyse pandas dataframes.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import lib.parse as parse
import lib.visualize as visu
import networkx as nx

pd.options.mode.chained_assignment = None

def analyse_simu(agent_response: Path, 
                analyse_dir: Path,
                graph_names,
                graph_colors,
                gifs = False) -> Tuple[str, int, str]:
    '''
        Analyse the respones of agents from a single simulation run.

        Args:
            agent_response: Path to the agent response CSV file.
            analyse_dir: Path to the directory where the analysis results will be saved.
            gifs: Boolean to determine if the figures should be created or not.

        Returns:
            graph_type: The type of graph used in the simulation.
            num_agents: The number of agents in the simulation.
            network_bias: The type of network bias used in the simulation.
    '''

    # Parse the file name
    num_agents, graph_type, network_bias = parse_file_path(agent_response)

    # Create the final result directories
    analyse_dir.mkdir(parents=True, exist_ok=True)

    # Parse the agent response
    agent_parsed_resp = parse.parse_output_mmlu(agent_response, analyse_dir / f'parsed_responses/{agent_response.name}.csv')
    network_responses_df = parse.get_network_responses(agent_parsed_resp, analyse_dir / f'network_responses/{agent_response.name}.csv')

    # Analyse the responses of this configuration
    results_path = analyse_dir / f'results/{agent_response.name}/'
    Path(results_path).mkdir(exist_ok=True)

    # Accuracy
    visu.accuracy_repartition(network_responses_df, f'{network_bias}', num_agents, results_path)

    # Consensus
    correct_consensus_df, incorrect_consensus_df = calculate_consensus_per_question(agent_parsed_resp, network_responses_df)
    visu.consensus_repartition(correct_consensus_df, incorrect_consensus_df, results_path,graph_colors)

    # Opinion changes
    opinion_changes = find_evolutions(agent_parsed_resp)
    visu.opinion_changes(opinion_changes, f"{graph_type}_{network_bias}", results_path, graph_names, graph_colors)

    # Neighbors
    calculate_proportion_neighbours_correct(agent_parsed_resp, graph_type, results_path)

    # Gifs
    if gifs:
        graphml_path = Path(f'experiment/data/{graph_type}_{network_bias}/{num_agents}.graphml')
        visu.created_gifs(agent_parsed_resp, graphml_path, results_path / f'{agent_response.name}/gifs/')

    return graph_type, num_agents, network_bias

def parse_file_path(dir_path : Path) -> Tuple[int, str, str]:
    '''
        Parse the file_path to determine the number of agents, the network bias the network type.

        Args:
            dir_path: Path to the agent response directory.

        Returns:
            number of agent: The number of agent in the simulation.
            graph_type: The type of graph used in the simulation.
            network_bias: The type of network bias used in the simulation.
    '''
    num_agents = 25 # fixed for this paper

    if "scale_free" in dir_path.name:
        graph_type = "scale_free"
    else:
        graph_type = dir_path.name

    network_bias = "unbiased"
    if 'scale_free' in dir_path.name:
        network_bias = "unbiased"
        if "incorrect" in dir_path.name and "hub" in dir_path.name:
            network_bias = "incorrect_hub"
        elif "incorrect" in dir_path.name and "edge" in dir_path.name:
            network_bias = "incorrect_edge"
        elif "correct" in dir_path.name and "hub" in dir_path.name:
            network_bias = "correct_hub"
        elif "correct" in dir_path.name and "edge" in dir_path.name:
            network_bias = "correct_edge"

    return num_agents, graph_type, network_bias

def rename_evolution(prev_correct, next_correct):
    """
        Return the type of evolution between two responses.

        Args:
            prev_correct: Boolean indicating if the previous response is correct.
            next_correct: Boolean indicating if the next response is correct.

        Returns:
            type: The type of evolution between the two responses.
    """
    type = None
    if bool(prev_correct):
        type = "C -> "
    else :
        type = "I -> "
    if bool(next_correct):
        type = type+"C"
    else :
        type = type+"I"
    return type

def find_evolutions(parsed_agent_response : pd.DataFrame) -> pd.DataFrame :
    """
    Return the list of changes that have occured in the simulation. For example, an agent changing is response from incorrect to correct.

    Args:
        parsed_agent_response: The parsed agent response dataframe.

    Returns:
        final_changes: The list of changes that have occured in the simulation.
    """
    df = parsed_agent_response.query("bias == 'unbiased'")
    final_changes : pd.DataFrame = None # Result

    # First we create a dataframe per round
    rounds: list[pd.DataFrame] = []
    for round in df['round'].unique():
        rounds.append(df.query(f'round == {round}'))
    
    # Then we join round n-1 and round n to record all changes
    for round in range(1,len(rounds)):
        prev, next = rounds[round-1], rounds[round]
        prev = prev.rename(columns={'correct' : 'prev_res', 'round': 'prev_round'})
        next = next.rename(columns={'correct' : 'cur_res'})
        changes = prev.merge(next, on=['network_number',
                                       'question_number',
                                       'repeat',
                                       'agent_id'])
        
        # We transform boolean into a string, "C" (correct) and "I" (Incorrect)
        changes['prev_res'] = changes['prev_res'].apply(lambda x : 'C' if x==True else 'I')
        changes['cur_res'] = changes['cur_res'].apply(lambda x : 'C' if x==True else 'I')

        changes['evolution'] = changes['prev_res'] + ' -> ' + changes['cur_res']
        changes = changes[['network_number',
                           'agent_id', 
                           'round',
                           'question_number',
                           'repeat',
                           'evolution']]

        # concatenation of all rounds results
        if final_changes is None:
            final_changes = changes
        else :
            final_changes = pd.concat([final_changes, changes], axis=0)

    return final_changes

def calculate_proportion_neighbours_correct(parsed_agent_response: pd.DataFrame, graph_type: str, final_res_path: Path) -> pd.DataFrame:
    """
    Calculate the proportion of neighbors that were correct in the previous round for each agent in unbiased responses,
    separately for each round, question number, and repeat, and merge this data back into the original DataFrame.
    
    Args:
        parsed_agent_response (pd.DataFrame): DataFrame containing agents' responses and metadata.
        graph_type (str): The type of graph to load (defines the directory of GraphML files).
        final_res_path (Path): The path to save the final results.
        
    Returns:
        pd.DataFrame: The original DataFrame with an added column for the proportion of correct neighbors from the previous round.
    """
    # Filter for unbiased responses
    df = parsed_agent_response.copy()
    df['agent_id_str'] = df['agent_id'].astype(str)  # Convert agent_id to string once

    new_data = []

    # Iterate over each unique combination of network number, round, question number, and repeat
    for (network_num, round_, question_number, repeat), df_group in df.groupby(['network_number', 'round', 'question_number', 'repeat']):
        graphml_path = Path(f'input/{graph_type}/{network_num}.graphml')
        G = nx.read_graphml(graphml_path)
        
        # Check if the previous round exists
        if round_ > 0:
            # Filter the DataFrame for the previous round
            df_previous_round = df.query(f"network_number == {network_num} and round == {round_ - 1} and question_number == {question_number} and repeat == {repeat}")

            # Iterate over each agent in the current group
            for agent_id_str in df_group['agent_id_str'].unique():
                if agent_id_str in G:
                    neighbors = list(G.neighbors(agent_id_str))
                    # Filter for neighbors' correctness in the previous round
                    df_neighbors_previous_round = df_previous_round[df_previous_round['agent_id_str'].isin(neighbors)]
                    proportion_correct_previous_round = df_neighbors_previous_round['correct'].mean() if not df_neighbors_previous_round.empty else None
                else:
                    proportion_correct_previous_round = None  # If agent_id not in graph, set proportion as None
                
                new_data.append({
                    'network_number': network_num,
                    'agent_id': int(agent_id_str),  # Convert back to int if necessary
                    'round': round_,
                    'question_number': question_number,
                    'repeat': repeat,
                    'proportion_neighbors_correct_previous_round': proportion_correct_previous_round
                })
    
    # Convert new data to DataFrame
    results_df = pd.DataFrame(new_data)
    # Specify columns to merge on, including new dimensions
    merge_cols = ['network_number', 'agent_id', 'round', 'question_number', 'repeat']
    # Merge the new data back into the original DataFrame
    df_final = pd.merge(parsed_agent_response, results_df, on=merge_cols, how='left')

    # Save csv
    df_final.to_csv(final_res_path / 'proportion_neighbors_correct_previous_round.csv', index=False)

    return df_final

def calculate_consensus_per_question(parsed_agent_response: pd.DataFrame, network_responses_df: pd.DataFrame):
    '''
    Returns consensus measure and Simpson consensus for each question in the parsed_agent_response DataFrame
    for questions which the system answered correctly and incorrectly.

    Args:
        parsed_agent_response: The parsed agent response dataframe.
        network_responses_df: The dataframe containing the system's response information.

    Returns:
        correct_consensus: The consensus measure and Simpson consensus for each question answered correctly.
        incorrect_consensus: The consensus measure and Simpson consensus for each question answered incorrectly.
    '''
    # Filter unbiased responses
    df = parsed_agent_response.query("bias == 'unbiased'")

    last_round = network_responses_df['round'].max()

    # Filter network_responses_df to include only correctly and incorrectly answered questions
    correct_questions = network_responses_df.query(f"round == {last_round} & correct == True")[['network_number', 'question_number', 'repeat']]
    incorrect_questions = network_responses_df.query(f"round == {last_round} & correct == False")[['network_number', 'question_number', 'repeat']]

    # Merge with the filtered parsed_agent_response to keep only the correct and incorrect questions
    correct_df = df.merge(correct_questions, on=['network_number', 'question_number', 'repeat'], how='inner')
    incorrect_df = df.merge(incorrect_questions, on=['network_number', 'question_number', 'repeat'], how='inner')

    # Function to compute consensus
    def compute_consensus(dataframe):
        # Calculate correct proportion for each question
        correct_prop = dataframe.groupby(['network_number', 'question_number', 'repeat'])['correct'].mean().reset_index()
        correct_prop = correct_prop.rename(columns={'correct': 'correct_prop'})

        # Calculate Simpson index
        counts = dataframe.groupby(['network_number', 'question_number', 'repeat', 'parsed_response']).size().reset_index(name='count')
        total_counts = counts.groupby(['network_number', 'question_number', 'repeat'])['count'].sum().reset_index(name='total')
        counts = counts.merge(total_counts, on=['network_number', 'question_number', 'repeat'])
        counts['proportion'] = counts['count'] / counts['total']
        counts['simpson'] = counts['proportion'] ** 2

        simpson = counts.groupby(['network_number', 'question_number', 'repeat'])['simpson'].sum().reset_index()

        final_consensus = pd.merge(correct_prop, simpson, on=['network_number', 'question_number', 'repeat'])

        return final_consensus

    # Compute consensus for correct and incorrect questions
    correct_consensus = compute_consensus(correct_df)
    incorrect_consensus = compute_consensus(incorrect_df)

    return correct_consensus, incorrect_consensus


