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

GRAPH_FILE_NAMES = ['fully_connected_network',
                'fully_disconnected_network',
                'scale_free_network',
                'random_network',
                'watts_strogatz_network']

GRAPH_PRETY_NAMES = ['Fully Connected Network',
                     'Fully Disconnected Network',
                     'Scale Free Network',
                     'Random Network',
                     'Watts-Strogatz Network']

AGENT_RESPONSES_REP = 'output/agent_responses/'
OUTPUT_ANALYSIS_REP = "output/analysis/"

class Analysis:
    """Meta data for a given graph and size"""

    def __init__(self, csv_path: str) -> None:
        # Type is a int, it is easier to compare
        self.type = file_name = GRAPH_FILE_NAMES.index(csv_path.split('/')[2])
        self.num_agents = int(csv_path.split('/')[3].split('.')[0])
        # The agent output is parsed and only the parsed file path is saved.
        self.path = parse_output_mmlu(csv_path, GRAPH_FILE_NAMES[self.type], self.num_agents)
        self.__accuracy = None
        self.__consensusPerQuestion = None
        self.__consensus = None
        self.__consensusSimpson = None
        self.__consensusPerQuestionWrongRes = None
        self.__consensusSimpsonWrongRes = None

    def setAccuracy(self) -> None:
        self.__accuracy = calculate_accuracy(self.path)

    def setConsensusPerQuestion(self) -> None:
        self.__consensusPerQuestion = calculate_consensus_per_question(self.path)

    def setConsensus(self) -> None:
        self.__consensus = self.getConsensusPerQuestion()['consensus'].mean()

    def setConsensusSimpson(self) -> None:
        '''
            The probability that two different agents randomly selected have the
        same responses.
        '''
        self.__consensusSimpson = self.getConsensusPerQuestion()['simpson'].mean()

    def setConsensusPerQuestionWrongRes(self) -> None:
        wrong_responses = filter_wrong_network_responses(self.path,
                                                        GRAPH_FILE_NAMES[self.type],
                                                        self.num_agents)
        self.__consensusPerQuestionWrongRes = calculate_consensus_per_question(wrong_responses)

    def setConsensusSimpsonWrongRes(self) -> None:
        '''
            Same as previously but only for the question on wich the network answer is wrong.
        '''
        self.__consensusSimpsonWrongRes = self.getConsensusPerQuestionWrongRes()['simpson'].mean()

    def getAccuracy(self) -> list[int]:
        # We check if it as already been computed
        if self.__accuracy is None:
            self.setAccuracy()
        return self.__accuracy

    def getConsensusPerQuestion(self) -> pd.DataFrame:
        # We check if it as already been computed
        if self.__consensusPerQuestion is None:
            self.setConsensusPerQuestion()
        return self.__consensusPerQuestion

    def getConsensus(self) -> float:
        # We check if it as already been computed
        if self.__consensus is None:
            self.setConsensus()
        return self.__consensus

    def getConsensusSimpson(self) -> float:
        # We check if it as already been computed
        if self.__consensusSimpson is None:
            self.setConsensusSimpson()
        return self.__consensusSimpson        

    def getConsensusPerQuestionWrongRes(self) -> pd.DataFrame:
        if self.__consensusPerQuestionWrongRes is None:
            self.setConsensusPerQuestionWrongRes()
        return self.__consensusPerQuestionWrongRes
    
    def getConsensusSimpsonWrongRes(self) -> float:
        if self.__consensusSimpsonWrongRes is None:
            self.setConsensusSimpsonWrongRes()
        return self.__consensusSimpsonWrongRes

    def ParseOpinionEvolution(self) -> pd.DataFrame :
        return find_evolutions(self.path)

    def plotDynamicEvolution(self) -> None :
        created_figs(self.path)

def created_figs(parsed_file_path: str, network_type: str) -> None:

    df = pd.read_csv(parsed_file_path, delimiter='|')

    df['response'] = df['response'].astype(str)

    # Now you can safely apply your function
    df['parsed_response'] = df['response'].apply(parse_response_mmlu)

    df['correct'] = df['parsed_response'] == df['correct_response']

    num_agents = df['agent_id'].unique().max() + 1

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

            round_num = int(round_) + 1

            # Add round number text on the plot
            plt.text(0.05, 0.95, f'Round: {round_num}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, facecolor='white'))

            # Save the plot for the current round
            image_path = f'output/figs/{network_type}/{num_agents}_q{question}_r{round_num}.png'
            directory = os.path.dirname(image_path)

            # Ensure the directory exists
            os.makedirs(directory, exist_ok=True)
            plt.savefig(image_path)
            plt.close('all')

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

    accuracies = []

    # Iterate through each round
    for round_number in df['round'].unique():
        round_df = df[df['round'] == round_number]

        correct_answers = 0
        total_questions = len(round_df['question_number'].unique())

        for question_number in round_df['question_number'].unique():
            question_df = round_df[round_df['question_number'] == question_number]

            # Find the most common response
            most_common_response = Counter(question_df['parsed_response']).most_common(1)[0][0]

            # Get the correct response (assuming it's the same for all rows of the same question)
            correct_answer = question_df['correct_response'].unique()
            assert(len(correct_answer)) # raise exception if all rows doesn't have the same correct answer
            correct_response = correct_answer[0]

            # Check if the most common response matches the correct response
            if most_common_response == correct_response:
                correct_answers += 1

        # Calculate the accuracy for the round
        accuracy_percentage = (correct_answers / total_questions) * 100
        accuracies.append({'round': round_number, 'accuracy': accuracy_percentage})

    # Convert the list of accuracies into a DataFrame
    accuracy_df = pd.DataFrame(accuracies)

    return accuracy_df

def find_evolutions(parsed_file_path) -> pd.DataFrame :
    opinion_evol_list = [] # list to be turned into a dataframe
    df = pd.read_csv(parsed_file_path, delimiter='|')

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

def calculate_consensus_per_question(parsed_file_path: str) -> pd.DataFrame :
    """
    """
    # Read the CSV file
    df = pd.read_csv(parsed_file_path, delimiter='|')

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
    final_consensus = pd.merge(final_consensus, simpson, on = 'question_number')#.set_index(to = 'question_number')
    return final_consensus[['question_number', 'parsed_response', 'consensus', 'simpson']]

def parse_output_mmlu(csv_file: str, res_file_name : str, num_agents: int) -> str:
    """
        Parse agent response to analyse which answer is correct and which is not.
    Save the result in OUTPUT_ANALYSIS_REP, adding '_parsed' res_file_name.
    Return the path of the result file.
    """
    df = pd.read_csv(csv_file, delimiter='|')

    # analysing responses to find the correct ones
    df['parsed_response'] = df['response'].apply(parse_response_mmlu)
    df['correct'] = df['parsed_response'] == df['correct_response']

    # If parsed response is not in the possible answers, we set parsed response as None
    df['parsed_response'] = df['parsed_response'].apply(lambda string: 
                                                        string if string in ['A', 'B', 'C', 'D'] 
                                                        else 'None')

    # removing useless columns
    df = df[['agent_id', 'round', 'question_number', 'parsed_response', 'correct_response', 'correct']]

    # saving data, creating the directory if it doesnot exist
    save_path = f'{OUTPUT_ANALYSIS_REP}parsed_agent_responses/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    save_path = f'{save_path}{res_file_name}_{num_agents}_parsed.csv'
    df.to_csv(save_path, mode='w', sep='|', index=False)

    return save_path

def filter_wrong_network_responses(parsed_reponses_csv: str, network_name: str, number_agent: int) -> str:
    """
        Filter the parsed response to keep only the data related to question where the network as globally
    given a wrong answer.
        The programm save the result as a .csv file and return the path. 
    """
    df =  pd.read_csv(parsed_reponses_csv, delimiter = '|')
    df.to_csv(Path(f'{OUTPUT_ANALYSIS_REP}parsed_agent_wrong_responses/step0.csv'), index=False)

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

    res_path = f'{OUTPUT_ANALYSIS_REP}parsed_agent_wrong_responses/{network_name}_{number_agent}.csv'
    res.to_csv(Path(res_path), index=False, mode='w', sep='|')
    return res_path

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

    # simpson consensus vs graph type for wrong answers
    plt.figure(figsize=(16, 9))
    for graph_type in range(len(analyses_list)):
        curve = []
        for analysis in analyses_list[graph_type]:
            curve.append([analysis.num_agents, analysis.getConsensusSimpsonWrongRes()])
        curve.sort(key = lambda point: point[0])
        X = [point[0] for point in curve]
        Y = [point[1] for point in curve]
        plt.plot(X, Y, label = GRAPH_PRETY_NAMES[graph_type])
    plt.title("Simpson Consensus vs Agent Graph Size and Graph Type For Wrong Answers")
    plt.legend()
    plt.xlabel('Number of agents (graph size)')
    plt.ylabel('Consensus (%)')
    plt.grid(True)
    plot_path = Path(OUTPUT_ANALYSIS_REP + 'simpson_consensus_vs_network_wrong_responses.png')
    plt.savefig(plot_path)
    plt.close('all') 