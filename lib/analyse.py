"""
    Contains programms to analyse pandas dataframes.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import lib.parse as parse
import lib.visualize as visu

pd.options.mode.chained_assignment = None

def analyse_simu(agent_response: Path, analyse_dir: Path, graph_names: dict[str, str], graph_colors: dict[str, str], gifs = False) -> Tuple[Path, str, int, str]:
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
    consensus_df = calculate_consensus_per_question(agent_parsed_resp)
    visu.consensus_repartition(consensus_df, graph_type, num_agents, final_res_path)

    # Opinion changes
    opinion_changes = find_evolutions(agent_parsed_resp)
    visu.opinion_changes(opinion_changes, network_bias, final_res_path, graph_names, graph_colors)

    # Figs
    if gifs:
        for network_num in range(3):
            graphml_path = Path(f'data/{graph_type}/{network_num}.graphml')
            visu.created_gifs(agent_parsed_resp, graphml_path, final_res_path / f'gifs/{network_num}/', network_num= network_num, graph_colors= graph_colors)

    # Wrong response consensus
    agent_parsed_wrong_responses = filter_wrong_responses(agent_parsed_resp,
                                                             network_responses_df)
    consensus_df = calculate_consensus_per_question(agent_parsed_wrong_responses)
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

    graph_type = "_".join(file_path.parent.name.split('_')[:-1])

    network_bias = "unbiased"
    if ("incorrect" in str(file_path)) and ("hub" in str(file_path)):
        network_bias = "incorrect_bias_hub"
    elif "incorrect" in str(file_path) and "edge" in str(file_path):
        network_bias = "incorrect_bias_edge"
    elif "correct" in str(file_path) and "hub" in str(file_path):
        network_bias = "correct_bias_hub"
    elif "correct" in str(file_path) and "edge" in str(file_path):
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

def calculate_consensus_per_question(parsed_agent_response: pd.DataFrame) -> pd.DataFrame :
    '''
        Returns consensus measure and Simpson consensus for each question in the parsed_agent_response dataFrame.

        Args:
            parsed_agent_response: The parsed agent response dataframe.

        Returns:
            final_consensus: The consensus measure and Simpson consensus for each question in the parsed_agent_response dataFrame.
    '''
    # Read the CSV file
    df = parsed_agent_response.query("bias == 'unbiased'")

    num_agent = df['agent_id'].unique().size
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

