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
import networkx as nx
import matplotlib.pyplot as plt
import imageio

def created_figs(file_path: str) -> None:

    df = pd.read_csv(file_path, delimiter='|')

    df['response'] = df['response'].astype(str)

    # Now you can safely apply your function
    df['parsed_response'] = df['response'].apply(parse_response_mmlu)

    df['correct'] = df['parsed_response'] == df['correct_response']

    network_type = file_path.split('/')[2]
    num_agents = file_path.split('/')[3].split('.')[0]

    graph = nx.read_graphml(f'data/{network_type}/{num_agents}.graphml')
    pos = nx.spring_layout(graph)

    rounds = df['round'].unique()
    questions = df['question_number'].unique()

    for question in questions:
        images = []  # To store paths of images for the current question
        for round_ in rounds:

            round_df = df[(df['round'] == round_) & (df['question_number'] == question)]
            color_map = ['green' if row['correct'] else 'red' for _, row in round_df.iterrows()]

            plt.figure(figsize=(10, 8))
            nx.draw(graph, pos=pos, node_color=color_map, with_labels=True, node_size=700)
            plt.title(f'Q{question} R{round_}')

            round_num = int(round_) + 1

            # Add round number text on the plot
            plt.text(0.05, 0.95, f'Round: {round_num}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, facecolor='white'))

            # Save the plot for the current round
            image_path = f'output/figs/{network_type}/{num_agents}_q{question}_r{round_num}.png'
            directory = os.path.dirname(image_path)

            # Ensure the directory exists
            os.makedirs(directory, exist_ok=True)
            plt.savefig(image_path)
            plt.close()

            images.append(image_path)

        # Create a GIF for the current question
        gif_path = f'output/figs/{network_type}/{num_agents}_q{question}.gif'
        with imageio.get_writer(gif_path, mode='I', fps=1, loop=0) as writer:
            for image_path in images:
                image = imageio.v3.imread(image_path)
                writer.append_data(image)
        
        # Optional: Remove the individual round images to clean up
        for image_path in images:
            os.remove(image_path)

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
    csv_files = glob.glob('output/agent_responses/**/*.csv', recursive=True)

    # Path for the results file
    results_file_path = output_analyse_path / 'results.csv'
    
    # Open the results file in write mode to ensure it's created or overwritten
    with open(results_file_path, 'w', newline='') as results_file:
        writer = csv.writer(results_file)
        # Write the headers
        writer.writerow(['network_type', 'accuracy'])
    
        # Now process each CSV file and append its results
        for file in csv_files:
            # Temporarily commented out for context
            # created_figs(file)

            file_name = os.path.basename(file).split('.')[0]
            network_type = file.split('/')[2]

            print(f"Processing {network_type}_{file_name}...")
            # Calculate the accuracy of group responses
            accuracy = calculate_accuracy(file)

            # Write the network type and accuracy to the results file
            writer.writerow([f"{network_type}_{file_name}", accuracy])  
