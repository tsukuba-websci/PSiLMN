import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import os
import csv
import glob
import pandas as pd
from collections import Counter
from lib.agent import parse_response_mmlu


# Function to calculate the accuracy of group responses
def calculate_accuracy(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, delimiter='|')
    
    # Group by question number and find the final round for each question
    final_rounds = df.groupby('question_number')['round'].max().reset_index()
    
    correct_answers = 0
    total_questions = final_rounds.shape[0]

    for _, row in final_rounds.iterrows():
        question_number = row['question_number']
        final_round = row['round']
        
        # Filter responses for the final round of this question
        final_responses = df[(df['question_number'] == question_number) & (df['round'] == final_round)].copy()

        # Correctly setting values to avoid SettingWithCopyWarning
        final_responses.loc[:, 'parsed_response'] = final_responses['response'].apply(parse_response_mmlu)

        # Find the most common response
        most_common_response = Counter(final_responses['parsed_response']).most_common(1)[0][0]
        
        # Get the correct response (assuming it's the same for all rows of the same question)
        correct_response = final_responses['correct_response'].iloc[0]
        
        # Check if the most common response matches the correct response
        if most_common_response == correct_response:
            correct_answers += 1
    
    # Calculate and return the accuracy percentage
    accuracy_percentage = (correct_answers / total_questions) * 100
    return accuracy_percentage

if __name__ == "__main__":
    
    # Create the new directory structure if it does not exist
    output_analyse_path = Path("output/analysis/")
    output_analyse_path.mkdir(parents=True, exist_ok=True)
        
    # Specify the path to your CSV file

    csv_files = glob.glob('output/agent_responses/*.csv')

    # Print the list of files
    for file in csv_files:
        
        file_name = os.path.basename(file).split('.')[0]

        # Calculate the accuracy of group responses
        accuracy = calculate_accuracy(file)

        with open('output/analysis/results.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            # Check if the file is empty to write headers
            if file.tell() == 0:
                writer.writerow(['file_name', 'accuracy'])
            writer.writerow([file_name, accuracy])
