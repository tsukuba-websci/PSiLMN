"""
    Contains programms to analyse pandas dataframes.
"""

import pandas as pd
from pathlib import Path

def find_evolutions(parsed_agent_response : pd.DataFrame) -> pd.DataFrame :
    """
        Return the list of changes that have occured in the simulation.
    For example, an agent changing is response from incorrect to correct.
    """
    opinion_evol_list = [] # list to be turned into a dataframe
    df = parsed_agent_response.query("bias == 'unbiased'")

    for network in df['network_number'].unique():
        for id in df['agent_id'].unique():
            for question in df['question_number'].unique():
                for round in [1,2]:
                    prev = df.query(f'network_number == {network} & agent_id == {id} & question_number == {question} & round == {round-1}')
                    assert len(prev.axes[0]) == 1
                    next = df.query(f'agent_id == {id} & question_number == {question} & round == {round}')
                    assert len(next.axes[0]) == 1

                    type = None
                    if bool(prev['correct'].values[0]):
                        type = "C -> "
                    else :
                        type = "I -> "
                    if bool(next['correct'].values[0]):
                        type = type+"C"
                    else :
                        type = type+"I"
                    opinion_evol_list.append([id, str(round), type])
    return pd.DataFrame(columns = ['network_number', 'agent_id', 'round', 'type'], data=opinion_evol_list)

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
    simpson = simpson.groupby(['question_number'])['simpson'].mean()
    simpson = simpson[['question_number','simpson']]

    # Finally, we join tables
    final_consensus = pd.merge(correct_prop, simpson, on = ['network_number', 'question_number'])

    return final_consensus
