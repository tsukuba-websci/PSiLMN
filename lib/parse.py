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

    # If parsed response is not in the possible answers, we set parsed response as X
    df['parsed_response'] = df['parsed_response'].apply(lambda string: 
                                                        string if string in ['A', 'B', 'C', 'D'] 
                                                        else 'X')
    # We harmonize bias column
    if "bias" in df.columns :
        df['bias'] = df['bias'].apply(lambda bias : 
                                      bias if bias in ['correct', 'incorrect'] 
                                      else "unbiased")
    else:
        df['bias'] = "unbiased"

    # Remove useless columns
    df = df[['network_number', 'agent_id', 'round', 'question_number', 'parsed_response', 'correct_response', 'correct', 'bias', 'repeat']]

    # Save the file
    df.to_csv(res_file_path, mode='w', sep='|', index=False)

    return df
