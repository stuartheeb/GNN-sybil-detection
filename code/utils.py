import numpy as np
from graphs import *


def dict_to_np_array(dict):
    """
    Convert a dictionary to a numpy array.
    :param dict: The dictionary to convert
    :return: A numpy array
    """
    # TODO  review : function was autocompleted with copilot
    keys = sorted(dict.keys())
    values = [dict[key] for key in keys]
    return np.array(values)


def np_array_to_dict(array):
    """
    Convert a numpy array to a dictionary.
    :param array: The numpy array to convert
    :return: A dictionary
    """
    # TODO  review : function was autocompleted with copilot
    return {i: array[i] for i in range(len(array))}


def relabel_graph_list_and_node_lists(graph_file_name: str,
                                      node_lists_file_names: [str],
                                      is_directed: bool,
                                      mapping: dict = None):
    # Read graph
    graph = EdgeListGraph(graph_file_name, is_directed=is_directed)
    nodes = graph.nodes_list()

    if mapping is None:
        # Go through graph node list and create dictionary reassigning IDs
        new_id = {}
        idx = 0
        for node in nodes:
            new_id[node] = idx
            idx += 1
        mapping = new_id

    # Rewrite graph .txt file with new IDs
    with open(graph_file_name.replace(".txt", "_post.txt"), 'w') as file:
        for (u, v) in graph.edges_list():
            file.write(str(mapping[u]) + ' ' + str(mapping[v]) + '\n')

    # Go through each node list and rewrite .txt file with new IDs
    for node_list_file_name in node_lists_file_names:
        with open(node_list_file_name, 'r') as read_file:
            lines = read_file.readlines()
            line_node_lists = []
            for i in range(len(lines)):
                line_node_lists.append([int(x) for x in lines[i].split()])

            with open(node_list_file_name.replace(".txt", "_post.txt"), 'w') as write_file:
                for node_list in line_node_lists:
                    write_file.write(' '.join(str(mapping[node]) for node in node_list) + '\n')


def mask_from_lists(nodes_lists: [[int]], labels: [int], no_label: int):
    max_idx = max(max(node_list) for node_list in nodes_lists)
    mask = np.full(max_idx + 1, no_label)
    for i in range(len(nodes_lists)):
        node_list = nodes_lists[i]
        for node in node_list:
            mask[node] = labels[i]
    return mask


def normalize_unit_interval(array: np.array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def write_string_to_file(string: str, file_name: str):
    with open(file_name, 'w') as file:
        file.write(string)


def plot_loss_curves(train_losses, val_losses, file_name: str):

    plt.figure(figsize=(12, 5))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(file_name)
    plt.close()
