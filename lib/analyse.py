"""
    Contains programms to analyse pandas dataframes.
"""

import pandas as pd
from collections import Counter

from typing import Tuple

def get_network_responses(parsed_agent_response: pd.DataFrame) -> pd.DataFrame:
    '''
        Return a dataFrame containing the accuracy for each round. 
    '''
    
    df =  parsed_agent_response
    
    # We count the number of responses of each type (A, B, C, D, X) for each question
    responses = df.groupby(['round',
                            'question_number', 
                            'parsed_response', 
                            'correct'], 
                            as_index = False).size()

    # We select the network answer at each question by selecting the most given answer at each question.
    responses = responses.sort_values(['round',
                                     'question_number', 
                                    'size'],
                                    ascending = False)
    return responses

    responses = responses.groupby(['round', 'question_number'],
                                    as_index= False).nth(0)
    
    return responses[['round','question_number','parsed_response','correct']]

def find_evolutions(parsed_agent_response : pd.DataFrame) -> pd.DataFrame :
    """
        Return the list of changes that have occured in the simulation.
    For example, an agent changing is response from incorrect to correct.
    """
    opinion_evol_list = [] # list to be turned into a dataframe
    df = parsed_agent_response

    for id in df['agent_id'].unique():
        for question in df['question_number'].unique():
            for round in [1,2]:
                prev = df.query(f'agent_id == {id} & question_number == {question} & round == {round-1}')
                assert len(prev.axes[0]) == 1
                next = df.query(f'agent_id == {id} & question_number == {question} & round == {round}')
                assert len(next.axes[0]) == 1

                # Here we determine wich type of evolution we have : 00, 01, 10 or 11,
                # with 0 wrong answer and 1 the good one (01 = from 1 to 0).
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
    return pd.DataFrame(columns = ['agent_id', 'round', 'type'], data=opinion_evol_list)

def calculate_consensus_per_question(parsed_agent_response: pd.DataFrame) -> pd.DataFrame :
    '''
        Return consensus measure and Simpson consensus for each question in the 
    parsed_agent_response dataFrame.
    '''
    # Read the CSV file
    df = parsed_agent_response

    num_agent = df['agent_id'].unique().size

    # We select only the correct responses in the last round and remove unecessary columns 
    final_consensus = df.query('correct & round == 2')[['question_number', 'correct']]

    # We count the proportion of agent with a correct answers for each question
    final_consensus = final_consensus.groupby(['question_number']).count().reset_index()
    final_consensus['consensus'] = final_consensus['correct'].apply(lambda correct : correct/num_agent)

    # Simpson consensus computation. This accuracy is computed as the probability that two answers
    # randomly selected are the same.
    simpson = df.query("round == 2").groupby(['question_number', 'parsed_response']).size().reset_index(name = 'count')
    simpson['simpson'] = simpson['count'].apply(lambda count : count/num_agent).apply(lambda p : p*p)
    simpson = simpson.groupby('question_number').sum().reset_index()[['question_number', 'parsed_response', 'simpson']]

    # Finally, we join tables
    final_consensus = pd.merge(final_consensus, simpson, on = 'question_number')

    return final_consensus
