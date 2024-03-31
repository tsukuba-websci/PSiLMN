"""
    Contains programms to analyse pandas dataframes.
"""

import pandas as pd
from pathlib import Path

def rename_evolution(prev_correct, next_correct):
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
        Return the list of changes that have occured in the simulation.
    For example, an agent changing is response from incorrect to correct.
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
        Return consensus measure and Simpson consensus for each question in the 
    parsed_agent_response dataFrame.
    '''
    # Read the CSV file
    df = parsed_agent_response.query("bias == 'unbiased'")

    num_agent = df['agent_id'].unique().size
    last_round = df['round'].unique().max()

    # We select only the correct responses in the last round and remove unecessary columns 
    correct_prop = df.query(f'round == {last_round}')#[['network_number', 'question_number', 'correct']]

    # We count the proportion of agent with a correct answers for each question
    correct_prop = correct_prop.groupby(['network_number', 'question_number'])['correct'].mean().reset_index()
    correct_prop.rename({'correct': 'Correct Agent Proportion'})

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
        Filter agent parsed rersponses to keep only the lines on which the response
    of the network is wrong.
    '''
    last_round = network_responses['round'].unique().max()
    wrong_responses = network_responses.query(f'correct == False & round == {last_round}')

    # We create new columns to concatenates the keys which will be used in isin()
    wrong_responses['key'] = wrong_responses['network_number'].apply(str) + '_' + wrong_responses['question_number'].apply(str) + '_' + wrong_responses['repeat'].apply(str)
    agent_parsed_responses['key'] = agent_parsed_responses['network_number'].apply(str) + '_' + agent_parsed_responses['question_number'].apply(str) + '_' + agent_parsed_responses['repeat'].apply(str)

    res = agent_parsed_responses[agent_parsed_responses['key'].isin(wrong_responses['key'])]

    return res.drop(columns='key')

