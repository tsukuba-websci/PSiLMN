"""
    Function used to parse csv files like the agent_response.
"""
import re
import pandas as pd

from glob import glob
from pathlib import Path
from random import shuffle
from typing import Optional

def parse_response_mmlu(response: str) -> Optional[str]:
    """
        Parse the string response for MMLU questions.
    Return Void if it can not find a response.
    """

    answer = None

    pattern = r'\(([a-zA-Z])\)'
    if response and not pd.isnull(response):
        matches = re.findall(pattern, response)

        for match_str in matches[::-1]:
            answer = match_str.upper()
            if answer:
                break

    return answer

def parse_output_mmlu(output_dir_path: Path, res_file_path: Path) -> pd.DataFrame:
    """
        Parse agent response csv files to analyse which answer is correct and which is not.
    Save the result in res_file_path. res_file_path should be in an existing repository.
    res_file_path should contain the file name and extension.
    Return the panda dataFrame.
    """
    # list of all csv files
    csv_files_to_parse = [Path(str_path) for str_path in glob(f'{str(output_dir_path)}/*.csv', recursive=False)]

    # Delete previous file and write headers
    with open(res_file_path, mode="w+") as f:
        f.write('network_number|agent_id|round|question_number|repeat|parsed_response|correct_response|correct|bias\n')
    
    for csv_file_to_parse in csv_files_to_parse:
        # load data from file name
        string_name_splited = csv_file_to_parse.name.split(sep = '.')[0].split(sep = '_')
        network_number, repeat = string_name_splited[2], string_name_splited[4]

        # load data from file
        df = pd.read_csv(csv_file_to_parse, delimiter='|')
        df = df.query("bias == 'none'")

        # Analyse responses to find the correct ones

        df['parsed_response'] = df['response'].apply(parse_response_mmlu)
        df['correct'] = df['parsed_response'] == df['correct_response']

        # If parsed response is not in the possible answers, we set parsed response as X
        df['parsed_response'] = df['parsed_response'].apply(lambda string: 
                                                            string if string in ['A', 'B', 'C', 'D'] 
                                                            else 'X')
        # We harmonize bias column
        if "correct" in csv_file_to_parse.name :
            df['bias'] = df['bias'].apply(lambda bias : 
                                        bias if bias in ['correct', 'incorrect'] 
                                        else "unbiased")
        else:
            df['bias'] = "unbiased"

        # Add repeat and network_number
        df['network_number'] = network_number
        df['repeat'] = repeat

        # Remove useless columns
        df = df[['network_number', 'agent_id', 'round', 'question_number', 'repeat', 'parsed_response', 'correct_response', 'correct', 'bias']]

        # Save the file
        df.to_csv(res_file_path, mode='a', sep='|', index=False, header=False)

    return pd.read_csv(res_file_path, delimiter='|')

def get_network_responses(parsed_agent_response: pd.DataFrame | Path, res_file_path: Path) -> pd.DataFrame:
    '''
        Return a dataFrame containing the agent response for each question and round, the
    DataFrame is also saved in save_path location. Save Path should constain the file name
    and extension (.csv).
    '''
    if isinstance(parsed_agent_response, Path):
        df = pd.read_csv(parsed_agent_response, sep = '|')
    elif isinstance(parsed_agent_response, pd.DataFrame):
        df = parsed_agent_response
    else:
        raise ValueError("parsed_agent_response should be a pandas DataFrame or a Path")
    
    # Biased node are not considered for the network answer.
    df = parsed_agent_response.query("bias == 'unbiased'")

    # We count the number of responses of each type (A, B, C, D, X) for each question
    responses: pd.DataFrame = df.groupby(['network_number',
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
    
    responses = responses[['network_number', 'round', 'question_number', 'repeat', 'parsed_response', 'correct']]
    responses.to_csv(res_file_path, mode='w', sep = '|', index=False)

    return responses
