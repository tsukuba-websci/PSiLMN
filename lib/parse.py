"""
    Function used to parse csv files like the agent_response.
"""
import re
import pandas as pd
from pathlib import Path

from typing import Optional

def parse_response_mmlu(response: str) -> Optional[str]:
    """
        Parse the string response for MMLU questions.
    Return Void if it can not find a response.
    """
    pattern = r'\(([a-zA-Z])\)'
    matches = re.findall(pattern, response)

    answer = None

    for match_str in matches[::-1]:
        answer = match_str.upper()
        if answer:
            break

    return answer

def parse_output_mmlu(csv_file_to_parse: Path, res_file_path: Path) -> pd.DataFrame:
    """
        Parse agent response csv file to analyse which answer is correct and which is not.
    Save the result in res_file_path. res_file_path should be in an existing repository.
    res_file_path should contain the file name and extension.
    Return the panda dataFrame.
    """
    df = pd.read_csv(csv_file_to_parse, delimiter='|')

    # Analyse responses to find the correct ones
    df['parsed_response'] = df['response'].apply(parse_response_mmlu)
    df['correct'] = df['parsed_response'] == df['correct_response']

    # If parsed response is not in the possible answers, we set parsed response as None
    df['parsed_response'] = df['parsed_response'].apply(lambda string: 
                                                        string if string in ['A', 'B', 'C', 'D'] 
                                                        else 'None')

    # Remove useless columns
    df = df[['agent_id', 'round', 'question_number', 'parsed_response', 'correct_response', 'correct']]

    # Save the file
    df.to_csv(res_file_path, mode='w', sep='|', index=False)

    return df

def filter_wrong_network_responses(parsed_reponses_csv: Path, res_file_path: Path) -> pd.DataFrame:
    """
        Filter the parsed response to keep only the data related to question where the network as globally
    given a wrong answer.
        Save the result in res_file_path. res_file_path should be in an existing repository.
    res_file_path should contain the file name and extension.
    Return the panda dataFrame. 
    """
    df =  pd.read_csv(parsed_reponses_csv, delimiter = '|')

    # We count the number of responses of each type (A, B, C, D, None) for each question
    network_wrong_responses = df.query('round == 2').groupby(['question_number', 
                                                                'parsed_response', 
                                                                'correct'],
                                                                as_index = False).size()

    # We select the network answer at each question by selecting the most given answer at each question.
    network_wrong_responses = network_wrong_responses.sort_values(['question_number', 
                                                                'size'],
                                                                ascending = False)
    network_wrong_responses = network_wrong_responses.groupby(['question_number'],
                                                                   as_index= False).nth(0)

    # We only keep the wrong responses
    network_wrong_responses = network_wrong_responses.query('not correct')['question_number']

    # By merging with the original data set, we only keep network wrong answers related data.
    res = pd.merge(df, network_wrong_responses, on="question_number")

    res.to_csv(res_file_path, index=False, mode='w', sep='|')
    return res
