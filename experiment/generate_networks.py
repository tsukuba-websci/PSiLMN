import matplotlib.pyplot as plt
from typing import List
import networkx as nx
import os

def save_networks(networks: List[nx.Graph], network_type:str, network_title: str) -> None:
    """
    Save a list of networks to disk.
    
    Args:
        networks (List[nx.Graph]): A list of networks to save.
        network_type (str): The type of network to save.
        network_title (str): The title of the network to save.
    """

    directory_path = f'data/{network_type}'
    os.makedirs(directory_path, exist_ok=True)
    for n, graph in zip(nodes_list, networks):
        file_path = f'{directory_path}/{n}.graphml'
        nx.write_graphml(graph, file_path)

        plt.figure(figsize=(8, 6))
        nx.draw_networkx(graph, node_size=50, with_labels=False)
        plt.title(f'{network_title} with {n} Nodes')
        file_path = f'{directory_path}/{n}.png'
        plt.savefig(file_path)
        plt.close()

if __name__ == "__main__":
    # Generate  networks with 10, 100, and 1000 nodes
    nodes_list = [10, 100, 1000]
    
    scale_free_networks = [nx.scale_free_graph(n).to_undirected() for n in nodes_list]
    for G in scale_free_networks:
        G.remove_edges_from(nx.selfloop_edges(G))

    watts_strogatz_networks = [nx.watts_strogatz_graph(n, k=4, p=0.1) for n in nodes_list]
    for G in watts_strogatz_networks:
        G.remove_edges_from(nx.selfloop_edges(G))

    random_networks = [nx.erdos_renyi_graph(n, p=0.15) for n in nodes_list]
    for G in random_networks:
        G.remove_edges_from(nx.selfloop_edges(G))

    fully_connected_networks = [nx.complete_graph(n) for n in nodes_list]
    for G in fully_connected_networks:
        G.remove_edges_from(nx.selfloop_edges(G))

    # Save all the networks
    save_networks(scale_free_networks, "scale_free_network", "Scale-free Network")
    save_networks(watts_strogatz_networks, "watts_strogatz_network", "Watts-Strogatz Network")
    save_networks(random_networks, "random_network", "Random Network")
    save_networks(fully_connected_networks, "fully_connected_network", "Fully Connected Network")