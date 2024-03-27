import networkx as nx
import numpy as np
import pandas as pd

from pathlib import Path

def get_hub_dist_df(graph_path: Path, hubs: list[str], max_length: int = None) -> pd.DataFrame:
    """
        For a given graph and a list of nodes, this program return a pd.dataFrame
    with the distance from each node to the closest node in the given list (hubs).
    """
    assert(max_length is None or max_length>0)
    graph =  nx.read_graphml(graph_path)
    for hub in hubs:
        short_path_length = nx.single_source_shortest_path_length(graph, hub)
        print(short_path_length)

