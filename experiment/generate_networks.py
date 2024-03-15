import matplotlib.pyplot as plt
from typing import List
import networkx as nx
import os
from xml.etree.ElementTree import ElementTree
import json

def graphml_to_json(filename: str, static: bool = False) -> None:
    """
    Convert a graphml file to json for use in the visualisation. Source: https://github.com/uskudnik/GraphGL/blob/master/examples/graphml-to-json.py
    
    Args:
        filename (str): The name of the graphml file to convert.
        static (bool): Whether to include static node attributes in the json file.
    """

    tree = ElementTree()
    with open(filename, "r") as file:
        tree.parse(file)

    graphml = {
        "graph": "{http://graphml.graphdrawing.org/xmlns}graph",
        "node": "{http://graphml.graphdrawing.org/xmlns}node",
        "edge": "{http://graphml.graphdrawing.org/xmlns}edge",
        "data": "{http://graphml.graphdrawing.org/xmlns}data",
        "label": "{http://graphml.graphdrawing.org/xmlns}data[@key='label']",
        "x": "{http://graphml.graphdrawing.org/xmlns}data[@key='x']",
        "y": "{http://graphml.graphdrawing.org/xmlns}data[@key='y']",
        "size": "{http://graphml.graphdrawing.org/xmlns}data[@key='size']",
        "r": "{http://graphml.graphdrawing.org/xmlns}data[@key='r']",
        "g": "{http://graphml.graphdrawing.org/xmlns}data[@key='g']",
        "b": "{http://graphml.graphdrawing.org/xmlns}data[@key='b']",
        "weight": "{http://graphml.graphdrawing.org/xmlns}data[@key='weight']",
        "edgeid": "{http://graphml.graphdrawing.org/xmlns}data[@key='edgeid']"
    }

    graph = tree.find(graphml.get("graph"))
    nodes = graph.findall(graphml.get("node"))
    links = graph.findall(graphml.get("edge"))

    out = {"nodes":[], "links":[]}  # Change to list for nodes

    for node in nodes:
        node_data = {
            "id": node.get("id"),
            "label": getattr(node.find(graphml.get("label")), "text", "")
        }
        if static:
            node_data.update({
                "size": float(getattr(node.find(graphml.get("size")), "text", 0)),
                "r": getattr(node.find(graphml.get("r")), "text", 0),
                "g": getattr(node.find(graphml.get("g")), "text", 0),
                "b": getattr(node.find(graphml.get("b")), "text", 0),
                "x": float(getattr(node.find(graphml.get("x")), "text", 0)),
                "y": float(getattr(node.find(graphml.get("y")), "text", 0))
            })
        out["nodes"].append(node_data)  # Append node data as dict to nodes list

    for edge in links:
        edge_data = {
            "source": edge.get("source"),
            "target": edge.get("target"),
        }
        edgeid = edge.find(graphml.get("edgeid"))
        if edgeid is not None:
            edge_data["edgeid"] = int(edgeid.text)
        out["links"].append(edge_data)

    network_type = filename.split("/")[1]
    outfilename =  os.path.basename(filename).rsplit(".", 1)[0] + ".json"

    output_path = f'data/{network_type}/{outfilename}'

    # Write the JSON data to the output file
    with open(output_path, "w") as outfile:
        json.dump(out, outfile, indent=4)


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
    nodes_list = [5, 25, 50]
    
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

    fully_disconnected_networks = [nx.empty_graph(n) for n in nodes_list]

    # Save all the networks
    save_networks(scale_free_networks, "scale_free_network", "Scale-free Network")
    save_networks(watts_strogatz_networks, "watts_strogatz_network", "Watts-Strogatz Network")
    save_networks(random_networks, "random_network", "Random Network")
    save_networks(fully_connected_networks, "fully_connected_network", "Fully Connected Network")
    save_networks(fully_disconnected_networks, "fully_disconnected_network", "Fully Disconnected Network")

    # Convert the graphml files to json for use in the visualisation
    networks = ["scale_free_network", "watts_strogatz_network", "random_network", "fully_connected_network", "fully_disconnected_network"]
    for network in networks:
        graphml_to_json(filename=f'data/{network}/5.graphml')
        graphml_to_json(filename=f'data/{network}/10.graphml')
        graphml_to_json(filename=f'data/{network}/25.graphml')
        graphml_to_json(filename=f'data/{network}/50.graphml')
        graphml_to_json(filename=f'data/{network}/100.graphml')
        graphml_to_json(filename=f'data/{network}/1000.graphml')