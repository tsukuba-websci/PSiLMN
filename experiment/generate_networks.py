import networkx as nx
import os

directory_path = 'data'
os.makedirs(directory_path, exist_ok=True)

# Generate scale-free networks with 10, 100, and 1000 nodes
nodes_list = [10, 100, 1000]
scale_free_networks = [nx.scale_free_graph(n) for n in nodes_list]

# Save the networks
for n, graph in zip(nodes_list, scale_free_networks):
    file_path = f'{directory_path}/scale_free_network_{n}_nodes.graphml'
    nx.write_graphml(graph, file_path)
