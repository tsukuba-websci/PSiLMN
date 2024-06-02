import matplotlib.pyplot as plt
import networkx as nx
import os
import json
from xml.etree.ElementTree import ElementTree

def graphml_to_json(filename: str, output_path: str, static: bool = False) -> None:
    """
    Convert a graphml file to json for use in the visualisation.
    
    Args:
        filename (str): The name of the graphml file to convert.
        output_path (str): The output path for the json file.
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

    outfilename = os.path.basename(filename).rsplit(".", 1)[0] + ".json"

    # Write the JSON data to the output file
    with open(os.path.join(output_path, outfilename), "w") as outfile:
        json.dump(out, outfile, indent=4)

def save_network(graph: nx.Graph, index: int, title: str, network_type: str) -> None:
    """
    Save a network to disk.
    
    Args:
        graph (nx.Graph): The network graph to save.
        index (int): The index of the network.
        title (str): The title of the network.
        network_type (str): The type of the network (e.g., scale_free, random, fully_connected, fully_disconnected).
    """

    location = f"input/{network_type}"
    os.makedirs(location, exist_ok=True)

    # Save the graph as graphml
    graphml_path = os.path.join(location, f'{index}.graphml')
    nx.write_graphml(graph, graphml_path)

    # Save the graph as json
    graphml_to_json(graphml_path, location)

    # Save graph as png using the load_and_plot_graph function
    png_path = os.path.join(location, f'{index}.png')
    load_and_plot_graph(graphml_path, png_path)

def create_random_network(size: int) -> nx.Graph:
    return nx.erdos_renyi_graph(size, 0.2)

def create_fully_connected_network(size: int) -> nx.Graph:
    return nx.complete_graph(size)

def create_fully_disconnected_network(size: int) -> nx.Graph:
    return nx.empty_graph(size)

def load_and_plot_graph(graphml_path, png_path):
    """
    Load a graph from a GraphML file and save its plot as a PNG file.
    
    Args:
        graphml_path (str): The path to the GraphML file.
        png_path (str): The output PNG file path.
    """
    # Load the graph from a GraphML file
    G = nx.read_graphml(graphml_path)

    # Draw the graph
    plt.figure(figsize=(8, 8))  # Set the figure size
    pos = nx.spring_layout(G, seed=42)  # Position nodes using the spring layout
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#377eb8', alpha=1)
    nx.draw_networkx_edges(G, pos, alpha=1)

    # Remove the axes
    plt.axis('off')

    # Save the plot to a PNG file
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()

if __name__ == "__main__":

    # Number of nodes in the network
    network_size = 25

    # Number of graphs to generate
    networks_per_size = 3
    
    # Create different types of networks
    scale_free_networks = [nx.scale_free_graph(network_size).to_undirected() for _ in range(networks_per_size)] 
    random_networks = [create_random_network(network_size) for _ in range(networks_per_size)]
    fully_connected_networks = [create_fully_connected_network(network_size) for _ in range(networks_per_size)]
    fully_disconnected_networks = [create_fully_disconnected_network(network_size) for _ in range(networks_per_size)]

    # Remove self loops from scale-free networks
    for G in scale_free_networks:
        G.remove_edges_from(nx.selfloop_edges(G))

    # Save all the networks
    for index, network in enumerate(scale_free_networks):
        save_network(graph=network, index=index, title=f"Scale-Free Network {index}", network_type="scale_free")
    for index, network in enumerate(random_networks):
        save_network(graph=network, index=index, title=f"Random Network {index}", network_type="random")
    for index, network in enumerate(fully_connected_networks):
        save_network(graph=network, index=index, title=f"Fully Connected Network {index}", network_type="fully_connected")
    for index, network in enumerate(fully_disconnected_networks):
        save_network(graph=network, index=index, title=f"Fully Disconnected Network {index}", network_type="fully_disconnected")
