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

    output_path = f'data/scale_free_network/{outfilename}'

    # Write the JSON data to the output file
    with open(output_path, "w") as outfile:
        json.dump(out, outfile, indent=4)


def save_network(graph: nx.Graph, index: int, title: str) -> None:
    """
    Save a list of networks to disk.
    
    Args:
        networks (List[nx.Graph]): A list of networks to save.
        network_type (str): The type of network to save.
        network_title (str): The title of the network to save.
    """

    location="data/scale_free_network"
    file_path = f'{location}/{index}.png'
    os.makedirs(location, exist_ok=True)

    # Save the graph as graphml
    file_path = f'{location}/{index}.graphml'
    nx.write_graphml(graph, file_path)

    # Save the graph as json
    graphml_to_json(file_path)

    # Save graph as png
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(graph, node_size=50, with_labels=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

if __name__ == "__main__":

    # Number of nodes in the network
    network_size = 25

    # Number of graphs to generate
    networks_per_size = 3
    
    # Create scale free networks
    scale_free_networks = [nx.scale_free_graph(network_size).to_undirected() for _ in range(networks_per_size)] 

    # Remove self loops
    for G in scale_free_networks:
        G.remove_edges_from(nx.selfloop_edges(G))

    # Save all the networks
    for index, network in enumerate(scale_free_networks):
        save_network(graph=network, index=index, title=f"Scale Free Network {index+1}")
