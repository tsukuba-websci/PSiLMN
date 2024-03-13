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
import seaborn as sns

GRAPH_TYPE = ['fully_connected_network',
            'scale_free_network',
            'random_network',
            'watts_strogatz_network']

# class OpinionEvolution:
#     """ 
#     Data structure to save an agent opinion between two round.
#     Round in [2, 3].
#     """
#     def __init__(self, previous : str, new : str, correct : str, round : int, agent_num: int) -> None:
#         self.previous_opinion = previous
#         self.new_opinion = new
#         self. correct_response = correct
#         self.round = round
#         self.agent = agent_num

#     def isAGoodChange(self) -> bool:
#         return self.response == self.new_opinion

class Analyse:
    """Meta data for a given graph and size"""

    def __init__(self, csv_path: str) -> None:
        # Type is a int, it is easier to compare
        self.type = file_name = GRAPH_TYPE.index(csv_path.split('/')[2])
        self.num_agents = int(csv_path.split('/')[3].split('.')[0])
        self.path = csv_path
        self.__accuracy = None

    def setAccuracy(self) -> None:
        self.__accuracy = calculate_accuracy(self.path)


    def getAccuracy(self) -> int:
        # We check if it as already been computed
        if not self.__accuracy:
            self.setAccuracy()
        return self.__accuracy

    def ParseOpinionEvolution(self) -> pd.DataFrame :
        return find_evolutions(self.path)

    def plotDynamicEvolution(self) -> None :
        created_figs(self.path)

def created_figs(file_path: str) -> None:

    df = pd.read_csv(file_path, delimiter='|')

    df['parsed_response'] = df['response'].apply(parse_response_mmlu)
    df['correct'] = df['parsed_response'] == df['correct_response']

    network_type = file_path.split('/')[2]
    num_agents = file_path.split('/')[3].split('.')[0]

    graph = nx.read_graphml(f'experiment/data/{network_type}/{num_agents}.graphml')
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

            # Add round number text on the plot
            plt.text(0.05, 0.95, f'Round: {round_}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, facecolor='white'))

            # Save the plot for the current round
            image_path = f'output/figs/{network_type}/{num_agents}_q{question}_r{round_+1}.png'
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

def find_evolutions(file_path) -> pd.DataFrame :
    opinion_evol_list = [] # list to be turned into a dataframe
    df = pd.read_csv(file_path, delimiter='|')

    df['parsed_response'] = df['response'].apply(parse_response_mmlu)
    df['correct'] = df['parsed_response'] == df['correct_response']

    for id in df['agent_id'].unique():
        for question in df['question_number'].unique():
            for round in [0,1]:
                prev = df.query(f'agent_id == {id} & question_number == {question} & round == {round}')
                assert len(prev.axes[0]) == 1
                next = df.query(f'agent_id == {id} & question_number == {question} & round == {round+1}')
                assert len(next.axes[0]) == 1

                # Here we determine wich type of evolution we have : 00, 01, 10 or 11,
                # with 0 wrong answer and 1 the good one (01 = from 1 to 0).
                type = None
                if bool(prev['correct'].values[0]):
                    type = "1"
                else :
                    type = "0"
                if bool(next['correct'].values[0]):
                    type = type+"1"
                else :
                    type = type+"0"
                opinion_evol_list.append([id, str(round + 1), type])
    return pd.DataFrame(columns = ['agent_id', 'round', 'type'], data=opinion_evol_list)   

if __name__ == "__main__":
    
    # Create the new directory structure if it does not exist
    output_analyse_path = "output/analysis/"
    Path(output_analyse_path).mkdir(parents=True, exist_ok=True)

    # Specify the path to your CSV file
    csv_files = glob.glob('output/agent_responses/**/*.csv', recursive=True)

    # analyses[i] contains the analyses relatives to a graph of type i
    analyses = [None] * len(GRAPH_TYPE)

    # Print the list of files
    for file in csv_files:
        print(file)
        current_analyse = Analyse(file) 
        
        # created_figs(file)

        # # Calculate the accuracy of group responses
        # accuracy = current_analyse.getAccuracy()

        # # # save accurracy
        # with open('output/analysis/accuracy.csv', 'a', newline='') as result_file:
        #     writer = csv.writer(result_file)
        #     # Check if the file is empty to write headers
        #     if result_file.tell() == 0:
        #         writer.writerow(['graph_type', 'size', 'accuracy'])
        #     writer.writerow([GRAPH_TYPE[current_analyse.type], current_analyse.num_agents, accuracy])

        # Parse the opinion evolution list and display it
        opinion_evol = current_analyse.ParseOpinionEvolution()
        plt.figure()
        sns.histplot(data=opinion_evol, x="round", hue="type", multiple="dodge", shrink=.8, stat="count")
        plt.grid()
        plt.xlabel("round number")
        plt.ylabel("number of agents")
        plt.title(f"opinion changes during the round for {current_analyse.num_agents} agents \nin a {GRAPH_TYPE[current_analyse.type]}")

        # Save the fig
        file_path = f'{output_analyse_path}/opinion_changes/{GRAPH_TYPE[current_analyse.type]}/'
        Path(file_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(f'{file_path}/{current_analyse.num_agents}.png'))
        plt.close()

        # # add analyse to the list for later
        # if not (analyses[current_analyse.type]):
        #     analyses[current_analyse.type] = []
        # analyses[current_analyse.type].append(current_analyse)

        # parse data to find changes

    # # accurracy comparison plot
    # plt.figure()
    # for graph_type in range(len(analyses)):
    #     curve = []
    #     for analyse in analyses[graph_type]:
    #         curve.append( (analyse.num_agents, analyse.getAccuracy()) )
    #     curve.sort(key = lambda point: point[0])
    #     # ploting this curve with seaborn
    #     X = [point[0] for point in curve]
    #     Y = [point[1] for point in curve]
    #     plt.plot(X, Y, label=GRAPH_TYPE[graph_type])
    # plt.title("accuracy vs graph size and graph type")
    # plt.legend()
    # plt.xlabel('Number of agents')
    # plt.ylabel('accuracy (%)')
    # plt.grid(True)
    # plot_path = Path(analyse_path + 'accuracy_vs_number_of_agents.png')
    # plt.savefig(plot_path)
    # plt.close()

    # opinion change analyse

