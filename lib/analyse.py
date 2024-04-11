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
                 graphs_path: Path,
                 graph_names: dict[str, str], 
                 graph_colors: dict[str, str], 
                 gifs = False) -> Tuple[Path, str, int, str]:
    '''
        Analyse the respones of agents from a single simulation run.

        Args:
            agent_response: Path to the agent response CSV file.
            analyse_dir: Path to the directory where the analysis results will be saved.
            figs: Boolean to determine if the figures should be created or not.

        Returns:
            final_res_path: Path to the directory where the analysis results are saved.
            graph_type: The type of graph used in the simulation.
            num_agents: The number of agents in the simulation.
            network_bias: The type of network bias used in the simulation.
    '''

    # Parse the file name
    num_agents, graph_type, network_bias = parse_file_path(agent_response)

    # Create the final result directory
    final_res_path = analyse_dir / f'{network_bias}/'
    final_res_path.mkdir(parents=True, exist_ok=True)

    # Parse the agent response
    agent_parsed_resp = parse.parse_output_mmlu(agent_response, final_res_path / 'agent_response_parsed.csv')
    network_responses_df = parse.get_network_responses(agent_parsed_resp, final_res_path / 'network_responses.csv')

    # Analyse the responses
    
    # Accuracy
    visu.accuracy_repartition(network_responses_df,
                              network_bias,
                              num_agents,
                              final_res_path)

    # Consensus
    consensus_df = calculate_consensus_per_question(agent_parsed_resp,network_bias)
    visu.consensus_repartition(consensus_df, graph_type, num_agents, final_res_path)

    # Opinion changes
    opinion_changes = find_evolutions(agent_parsed_resp)
    visu.opinion_changes(opinion_changes, network_bias, final_res_path, graph_names, graph_colors)

    # Get correctness by proportion of correct neighbours
    calculate_proportion_neighbours_correct(agent_parsed_resp, graphs_path, final_res_path)

    # Figs
    if gifs:
        for network_num in range(3):
            graphml_path = Path(f'data/{graph_type}/{network_num}.graphml')
            visu.created_gifs(agent_parsed_resp, graphml_path, final_res_path / f'gifs/{network_num}/', network_num= network_num, graph_colors= graph_colors)

    # Wrong response consensus
    agent_parsed_wrong_responses = filter_wrong_responses(agent_parsed_resp,
                                                             network_responses_df)
    consensus_df = calculate_consensus_per_question(agent_parsed_wrong_responses, network_bias)
    visu.consensus_repartition(consensus_df,
                               f'{graph_type} (wrong ansers only)',
                               num_agents,
                               final_res_path,
                               wrong_response= True)

    return final_res_path, graph_type, num_agents, network_bias

def parse_file_path(file_path : Path) -> Tuple[int, str, str]:
    '''
        Parse the file_path to determine the number of agents, the network bias the network type.

        Args:
            file_path: Path to the agent response CSV file.

        Returns:
            num_agents: The number of agents in the simulation.
            graph_type: The type of graph used in the simulation.
            network_bias: The type of network bias used in the simulation.
    '''
    num_agents = int(file_path.name.split('.')[0])

    graph_type = "scale_free_network"

    network_bias = "unbiased"
    if ("incorrect" in str(file_path)) and ("hub" in str(file_path)):
        network_bias = "incorrect_bias_hub"
    elif ("incorrect" in str(file_path)) and ("edge" in str(file_path)):
        network_bias = "incorrect_bias_edge"
    elif ("correct" in str(file_path)) and ("hub" in str(file_path)):
        network_bias = "correct_bias_hub"
    elif ("correct" in str(file_path)) and ("edge" in str(file_path)):
        network_bias = "correct_bias_edge"

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

def calculate_proportion_neighbours_correct(parsed_agent_response: pd.DataFrame, graphml_file: Path, final_res_path: Path) -> pd.DataFrame:
    """
    Calculate the proportion of neighbors that were correct in the previous round for each agent in unbiased responses,
    separately for each round, question number, and repeat, and merge this data back into the original DataFrame.
    
    Args:
        parsed_agent_response (pd.DataFrame): DataFrame containing agents' responses and metadata.
        graphml_path (Path): The file containing graphs to load (defines the directory of GraphML files).
        final_res_path (Path): The path to save the final results.
        
    Returns:
        pd.DataFrame: The original DataFrame with an added column for the proportion of correct neighbors from the previous round.
    """
    # Filter for unbiased responses
    df = parsed_agent_response.query("bias == 'unbiased'").copy()
    df['agent_id_str'] = df['agent_id'].astype(str)  # Convert agent_id to string once

    new_data = []

    # Iterate over each unique combination of network number, round, question number, and repeat
    for (network_num, round_, question_number, repeat), df_group in df.groupby(['network_number', 'round', 'question_number', 'repeat']):
        graphml_path = graphml_file / f'{network_num}.graphml'
        G = nx.read_graphml(graphml_path)
        
        # Check if the previous round exists
        if round_ > 0:
            # Filter the DataFrame for the previous round
            df_previous_round = df.query(f"network_number == {network_num} and round == {round_ - 1} and question_number == {question_number} and repeat == {repeat}")

            # Iterate over each agent in the current group
            for agent_id_str in df_group['agent_id_str'].unique():
                if agent_id_str in G:
                    neighbors = list(G.neighbors(agent_id_str))
                    # Find the agent previous answer
                    previous_resp = df_previous_round.query(f"agent_id == {agent_id_str}")['correct'].iloc[0]

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
                    'previous_response': previous_resp,
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

def neighbours_accuracy_tables(parsed_agent_response: pd.DataFrame) -> None:
    # filter round not equal to 1
    df = parsed_agent_response[parsed_agent_response['round'] != 0] 

    proportions = [[0., 0.1], [0.1, 0.45], [0.45, 0.55], [0.55, 0.95], [0.95, 1.1]]
    count_table = pd.DataFrame(data = [True, False], columns=['previous_opinion'])
    proba_table = pd.DataFrame(data = [True, False], columns=['previous_opinion'])

    for [min, max] in proportions: 
        sample = df.query(f'proportion_neighbors_correct_previous_round >= {min} & proportion_neighbors_correct_previous_round < {max}')
        count = sample.groupby('previous_response').size()
        proba = sample.groupby(['previous_response'])['correct'].mean()
        count_table[f'{min}_{max}'] = count
        proba_table[f'{min}_{max}'] = proba
    return count_table, proba_table

def calculate_consensus_per_question(parsed_agent_response: pd.DataFrame, network_bias: str) -> pd.DataFrame :
    '''
        Returns consensus measure and Simpson consensus for each question in the parsed_agent_response dataFrame.

        Args:
            parsed_agent_response: The parsed agent response dataframe.

        Returns:
            final_consensus: The consensus measure and Simpson consensus for each question in the parsed_agent_response dataFrame.
    '''

    biased = True if 'unbiased' not in network_bias else False

    # Read the CSV file
    df = parsed_agent_response.query("bias == 'unbiased'")

    num_agent = 23 if biased else 25
    last_round = df['round'].unique().max()

    # We select only the correct responses in the last round and remove unecessary columns 
    correct_prop = df.query(f'round == {last_round}')#[['network_number', 'question_number', 'correct']]

    # We count the proportion of agent with a correct answers for each question
    correct_prop = correct_prop.groupby(['network_number', 'question_number'])['correct'].mean().reset_index()
    correct_prop = correct_prop.rename(columns={'correct': 'correct_prop'})
    
    # Simpson consensus computation. This accuracy is computed as the probability that two answers
    # randomly selected are the same.
    simpson = df.query(f"round == {last_round}").groupby(['network_number', 
                                                            'question_number', 
                                                            'parsed_response',
                                                            'repeat']).size().reset_index(name = 'count')
    simpson['simpson'] = simpson['count'].apply(lambda count : count/num_agent).apply(lambda p : p*p)
    simpson = simpson.groupby(['network_number', 'question_number', 'repeat']).sum().reset_index()

    # average on 'repeat'
    simpson = simpson.groupby(['network_number', 'question_number'])['simpson'].mean().reset_index()
    simpson = simpson[['network_number', 'question_number','simpson']]

    # Finally, we join tables
    final_consensus = pd.merge(correct_prop, simpson, on = ['network_number', 'question_number'])

    return final_consensus

def filter_wrong_responses(agent_parsed_responses: pd.DataFrame,
                           network_responses: pd.DataFrame) -> pd.DataFrame:
    '''
        Filter agent parsed rersponses to keep only the lines on which the response of the network is wrong.

        Args:
            agent_parsed_responses: The parsed agent response dataframe.
            network_responses: The network response dataframe.

        Returns:
            res: The filtered agent parsed responses dataframe.
    '''
    last_round = network_responses['round'].unique().max()
    wrong_responses = network_responses.query(f'correct == False & round == {last_round}')

    # We create new columns to concatenates the keys which will be used in isin()
    wrong_responses.loc[:, 'key'] = wrong_responses['network_number'].astype(str) + '_' + wrong_responses['question_number'].astype(str) + '_' + wrong_responses['repeat'].astype(str)
    agent_parsed_responses.loc[:, 'key'] = agent_parsed_responses['network_number'].apply(str) + '_' + agent_parsed_responses['question_number'].apply(str) + '_' + agent_parsed_responses['repeat'].apply(str)

    res = agent_parsed_responses[agent_parsed_responses['key'].isin(wrong_responses['key'])]

    return res.drop(columns='key')

