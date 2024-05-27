import matplotlib.pyplot as plt
import networkx as nx

def load_and_plot_graph(graphml_path, png_path):
    # Load the graph from a GraphML file
    G = nx.read_graphml(graphml_path)


    try:
        diameter = nx.diameter(G)
        print(f"The diameter of the graph is: {diameter}")
    except nx.NetworkXError as e:
        print(f"Error calculating diameter: {e}")
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

# Example usage
graphml_path = '/Users/ciaran/dev/MultiAgentNetworkSimulation/experiment/data/scale_free_network/0.graphml'  # Replace with your GraphML file path
png_path = 'tmp.png'  # The output PNG file name
load_and_plot_graph(graphml_path, png_path)

print(f"Graph image saved to {png_path}")
