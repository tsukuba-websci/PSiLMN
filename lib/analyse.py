"""
    Contains programms to analyse pandas dataframes.
"""

import pandas as pd
from collections import Counter
from random import shuffle

def get_network_responses(parsed_agent_response: pd.DataFrame) -> pd.DataFrame:
    '''
        Return a dataFrame containing the agent response for each question and round. 
    '''
    
    df = parsed_agent_response
    df = parsed_agent_response.query("bias == 'unbiased'")

    # We count the number of responses of each type (A, B, C, D, X) for each question
    responses = df.groupby(['network_number',
                            'round',
                            'question_number', 
                            'parsed_response', 
                            'correct',
                            'repeat'], 
                            as_index = False).size()

    # We add a random value at each line. This allow us to randomly select the answer if to answers have
    # been given by the same number of agents.
    random_vect = list(range(responses.shape[0]))
    shuffle(random_vect)
    responses['rd_number'] = random_vect

    # We select the network answer at each question by selecting the most given answer at each question.
    responses = responses.sort_values(['network_number',
                                       'round',
                                     'question_number', 
                                    'size',
                                    'rd_number',
                                    'repeat'],
                                    ascending = False)

    responses = responses.groupby(['network_number',
                                   'round',
                                    'question_number',
                                    'repeat']).nth(0)
    return responses[['network_number', 'round', 'question_number', 'parsed_response', 'correct', 'repeat']]

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
    final_consensus = df.query(f'correct & round == {last_round}')[['network_number', 'question_number', 'correct']]

    # We count the proportion of agent with a correct answers for each question
    final_consensus = final_consensus.groupby(['network_number', 'question_number']).count().reset_index()
    final_consensus['consensus'] = final_consensus['correct'].apply(lambda correct : correct/num_agent)

    # Simpson consensus computation. This accuracy is computed as the probability that two answers
    # randomly selected are the same.
    print(df['question_number'])
    simpson = df.query(f"round == {last_round}").groupby(['network_number', 
                                                            'question_number', 
                                                            'parsed_response']).size().reset_index(name = 'count')
    simpson['simpson'] = simpson['count'].apply(lambda count : count/num_agent).apply(lambda p : p*p)
    simpson = simpson.groupby('network_number', 'question_number').sum().reset_index()[['network_number',
                                                                                        'question_number', 
                                                                                        'parsed_response', 
                                                                                        'simpson']]

    # Finally, we join tables
    final_consensus = pd.merge(final_consensus, simpson, on = ['network_number', 'question_number'])

    return final_consensus
