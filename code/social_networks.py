from graphs import *
from attacks import *
from utils import *

import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class SocialNetwork:
    def __init__(self, network: Graph = None,
                 honest_nodes: [int] = None,
                 sybil_nodes: [int] = None,
                 name: str = "SocialNetwork",
                 reciprocal: bool = False) -> None:

        self.reciprocal = reciprocal

        self.network = None
        self.undirected_network = None
        self.directed_network = None
        if network is not None:
            self.set_network(network)

        self.honest_nodes = None
        self.sybil_nodes = None
        self.update_known_nodes(honest_nodes=honest_nodes, sybil_nodes=sybil_nodes)

        self.train_honest_nodes = None
        self.train_sybil_nodes = None
        self.test_honest_nodes = None
        self.test_sybil_nodes = None
        # self.train_honest_nodes, self.train_sybil_nodes, self.test_honest_nodes, self.test_sybil_nodes = self.train_test_split()

        self.name = name

    def __str__(self):
        # if self.network is None:
        #    return self.name + ": (None)"
        return f"{self.name}:\nUndirected version: {'None' if self.undirected_network is None else self.undirected_network}\nDirected version: {'None' if self.directed_network is None else self.directed_network}\nTrain:\t{len(self.train_honest_nodes)} (honest)\t/ {len(self.train_sybil_nodes)} (sybil)\nTest:\t{len(self.test_honest_nodes)} (honest)\t/ {len(self.test_sybil_nodes)} (sybil)\nTotal:\t{len(self.honest_nodes)} (honest)\t/ {len(self.sybil_nodes)} (sybil)"

    def set_network(self, network: Graph, reciprocal: bool = None):
        if reciprocal is None:
            reciprocal = self.reciprocal

        self.network = network
        if network.is_directed():
            # print("Directed network detected. Creating undirected network.")
            self.directed_network = network
            self.undirected_network = network.undirected_graph_copy(reciprocal=reciprocal)
        else:
            # print("Undirected network detected. Creating directed network.")
            self.undirected_network = network
            self.directed_network = network.directed_graph_copy()

    def relabel_network(self, mapping: dict = None):
        mapping = self.network.relabel_nodes(mapping)
        self.set_network(self.network)

        count = 0
        for nodes_list in [self.honest_nodes, self.sybil_nodes, self.train_honest_nodes, self.train_sybil_nodes,
                           self.test_honest_nodes, self.test_sybil_nodes]:
            if nodes_list is not None:
                new_list = []
                for i in range(len(nodes_list)):
                    if nodes_list[i] in mapping.keys():
                        new_list.append(mapping[nodes_list[i]])
                if count == 0:
                    self.honest_nodes = new_list
                elif count == 1:
                    self.sybil_nodes = new_list
                elif count == 2:
                    self.train_honest_nodes = new_list
                elif count == 3:
                    self.train_sybil_nodes = new_list
                elif count == 4:
                    self.test_honest_nodes = new_list
                elif count == 5:
                    self.test_sybil_nodes = new_list
            count += 1

        # print(f"len(honest_nodes) = {len(self.honest_nodes)}")
        # print(f"len(sybil_nodes) = {len(self.sybil_nodes)}")

    def add_edges(self, edges: [(int, int)]):
        self.network.graph.add_edges_from(edges)
        self.set_network(self.network)

    def fraction_of_sybils(self):
        if self.sybil_nodes is None:
            return 0.0
        return len(self.sybil_nodes) / self.network.num_nodes()

    def update_known_nodes(self,
                           honest_nodes: list[int] = None,
                           sybil_nodes: list[int] = None,
                           force_update: bool = False,
                           update_labels: bool = False):
        if (force_update or honest_nodes != self.honest_nodes) and honest_nodes is not None:
            self.honest_nodes = honest_nodes

        if (force_update or sybil_nodes != self.sybil_nodes) and sybil_nodes is not None:
            self.sybil_nodes = sybil_nodes

        # TODO maybe use update_labels flag to update labels of nodes in the graph (properties)
        # e.g. in order to draw the graph with these properties (e.g. colors indicating honest/sybil nodes)

    def train_test_split(self, train_fraction: float = None,
                         train_fraction_honest: float = 0.1,
                         train_fraction_sybil: float = 0.1):
        if train_fraction is not None:
            train_fraction_honest = train_fraction
            train_fraction_sybil = train_fraction

        self.train_honest_nodes = random.sample(self.honest_nodes,
                                                k=int(train_fraction_honest * len(self.honest_nodes)))
        self.train_sybil_nodes = random.sample(self.sybil_nodes, k=int(train_fraction_sybil * len(self.sybil_nodes)))

        self.test_honest_nodes = list(set(self.honest_nodes) - set(self.train_honest_nodes))
        self.test_sybil_nodes = list(set(self.sybil_nodes) - set(self.train_sybil_nodes))

        return self.train_honest_nodes, self.train_sybil_nodes, self.test_honest_nodes, self.test_sybil_nodes

    def get_train_test_split(self):
        if self.train_honest_nodes is None or self.train_sybil_nodes is None or self.test_honest_nodes is None or self.test_sybil_nodes is None:
            return self.train_test_split()
        return self.train_honest_nodes, self.train_sybil_nodes, self.test_honest_nodes, self.test_sybil_nodes

    def degree_distributions(self, directory: str = "output", file_name: str = None, network_num_bins: int = -1):
        if file_name is None:
            file_name = "degree_distribution.pdf"

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.network.degree_distribution(directory=directory,
                                         file_name=file_name,
                                         positive_nodes=self.sybil_nodes,
                                         coloring=True,
                                         num_bins=network_num_bins)
        Graph(graph=self.network.graph.subgraph(self.honest_nodes)).degree_distribution(directory=directory,
                                                                                        file_name=f"{file_name.split(sep='.pdf')[0]}_honests.pdf")
        Graph(graph=self.network.graph.subgraph(self.sybil_nodes)).degree_distribution(directory=directory,
                                                                                       file_name=f"{file_name.split(sep='.pdf')[0]}_sybils.pdf")

    def analyse_attack_edges(self):
        edges = self.network.edges_list()
        print(f"Total number of edges: {len(edges)}")
        honest_sybil_count = 0
        sybil_honest_count = 0
        i = 0
        for (u, v) in edges:
            if u in self.honest_nodes and v in self.sybil_nodes:
                honest_sybil_count += 1
            elif u in self.sybil_nodes and v in self.honest_nodes:
                sybil_honest_count += 1
            i += 1

        if self.network.is_undirected():
            total = honest_sybil_count + sybil_honest_count
            attack_edges_per_sybil = total / len(self.sybil_nodes)
            print(f"Total number of attack edges: {total}")
            print(f"Attack edges per sybil: {attack_edges_per_sybil}")
        else:
            honest_sybil_per_honest = honest_sybil_count / len(self.honest_nodes)
            sybil_honest_per_sybil = sybil_honest_count / len(self.sybil_nodes)
            print(f"Number of honest -> sybil edges: {honest_sybil_count} ({honest_sybil_per_honest} per honest)")
            print(f"Number of sybil -> honest edges: {sybil_honest_count} ({sybil_honest_per_sybil} per sybil)")

    def save_network_graph(self, file_name="output/graph.pdf") -> None:
        self.update_known_nodes(update_labels=True)  # This will only update labels, not the node sets
        self.network.save_graph(file_name=file_name)

    def write_gexf(self, file_name="output/social_network.gexf") -> None:
        self.network.write_gexf(file_name=file_name)

    def write_graph_train_test_split(self, directory: str, file_type: str = "edge_list",
                                     both_edges_for_undirected: bool = False,
                                     train_honest_nodes: [int] = None,
                                     train_sybil_nodes: [int] = None,
                                     test_honest_nodes: [int] = None,
                                     test_sybil_nodes: [int] = None):

        if train_honest_nodes is None or train_sybil_nodes is None or test_honest_nodes is None or test_sybil_nodes is None:
            train_honest_nodes = self.train_honest_nodes
            train_sybil_nodes = self.train_sybil_nodes
            test_honest_nodes = self.test_honest_nodes
            test_sybil_nodes = self.test_sybil_nodes

        if train_honest_nodes is None or train_sybil_nodes is None or test_honest_nodes is None or test_sybil_nodes is None:
            raise Exception("Train and test nodes not set")

        if not os.path.exists(directory):
            os.makedirs(directory)

        if file_type == "edge_list":
            self.network.write_edge_list(file_name=f"{directory}/graph.txt",
                                         both_edges_for_undirected=both_edges_for_undirected)
        elif file_type == "gexf":
            self.network.write_gexf(file_name=f"{directory}/graph.gexf")
        else:
            raise Exception("Unknown file type")

        Graph.write_node_lists_to_file(node_lists=[train_honest_nodes, train_sybil_nodes],
                                       file_name=f"{directory}/train.txt")
        Graph.write_node_lists_to_file(node_lists=[test_honest_nodes, test_sybil_nodes],
                                       file_name=f"{directory}/test.txt")
        Graph.write_node_lists_to_file(node_lists=[train_honest_nodes + test_honest_nodes,
                                                   train_sybil_nodes + test_sybil_nodes],
                                       file_name=f"{directory}/test_full.txt")

    def write_graph_honest_sybil(self, directory: str, file_type: str = "edge_list"):
        if not os.path.exists(directory):
            os.makedirs(directory)

        if file_type == "edge_list":
            self.network.write_edge_list(file_name=f"{directory}/graph.txt")
        elif file_type == "gexf":
            self.network.write_gexf(file_name=f"{directory}/graph.gexf")
        else:
            raise Exception("Unknown file type")

        Graph.write_node_lists_to_file(node_lists=[self.honest_nodes],
                                       file_name=f"{directory}/honest.txt")
        Graph.write_node_lists_to_file(node_lists=[self.sybil_nodes],
                                       file_name=f"{directory}/sybil.txt")


class SocialNetworkFromRegions(SocialNetwork):
    def __init__(self,
                 honest_region: Graph = None,
                 sybil_region: Graph = None,
                 num_attack_edges=None,
                 attack: Attack = None,
                 name: str = "SocialNetworkFromRegions") -> None:
        super().__init__(honest_nodes=honest_region.nodes_list(),
                         sybil_nodes=sybil_region.nodes_list(),
                         name=name)

        self.honest_region = honest_region
        self.sybil_region = sybil_region

        self._check_regions()
        self._combine_regions()

        if isinstance(num_attack_edges, int):
            self.num_attack_edges = [num_attack_edges, 0, 0]
        elif isinstance(num_attack_edges, list) and all(isinstance(x, int) for x in num_attack_edges) and len(
                num_attack_edges) == 3:
            self.num_attack_edges = num_attack_edges
        elif num_attack_edges is None:
            avg_degree_honest_region = honest_region.average_degree()
            self.num_attack_edges = [0.5 * avg_degree_honest_region * sybil_region.num_nodes(), 0, 0]
        else:
            raise Exception("Invalid number of attack edges provided")

        if attack is None:
            self.attack = RandomAttack()
        else:
            self.attack = attack

    def _check_regions(self) -> None:
        if self.honest_region.is_sybil:
            raise Exception("Honest region graph is sybil")

        if not self.sybil_region.is_sybil:
            raise Exception("Sybil region graph is not sybil")

    def _combine_regions(self):
        network = nx.disjoint_union(self.honest_region.graph, self.sybil_region.graph)
        self.set_network(Graph(graph=network, is_network=True))

        n_G = self.honest_region.graph.number_of_nodes()
        sybil_nodes = [i + n_G for i in self.sybil_nodes]
        self.update_known_nodes(honest_nodes=self.honest_nodes, sybil_nodes=sybil_nodes)

    def perform_attack(self, num_attack_edges: [int] = None, attack: Attack = None):
        if num_attack_edges is None:
            num_attack_edges = self.num_attack_edges
        if attack is None:
            attack = self.attack

        attack.perform_attack(social_network=self, num_attack_edges=num_attack_edges)


class SocialNetworkFromGraphAndLists(SocialNetwork):
    def __init__(self,
                 network: Graph,
                 honest_nodes: [int],
                 sybil_nodes: [int]):
        super().__init__(network=network,
                         honest_nodes=honest_nodes,
                         sybil_nodes=sybil_nodes)


class SocialNetworkFromRegionFiles(SocialNetwork):
    def __init__(self,
                 network_file_name: str,
                 honest_nodes_file_name: str,
                 sybil_nodes_file_name: str,
                 network_file_type: str = "edge_list",
                 is_directed: bool = False):
        super().__init__(
            network=Graph.get_graph_from_file(network_file_name, network_file_type, is_directed=is_directed),
            honest_nodes=Graph.get_node_list_from_file(honest_nodes_file_name),
            sybil_nodes=Graph.get_node_list_from_file(sybil_nodes_file_name))


class SocialNetworkFromTrainTestSets(SocialNetwork):
    def __init__(self,
                 network: Graph,
                 train_honest_nodes: [int],
                 train_sybil_nodes: [int],
                 test_honest_nodes: [int],
                 test_sybil_nodes: [int],
                 name: str = "Network",
                 reciprocal: bool = False):
        super().__init__(network=network,
                         honest_nodes=train_honest_nodes + test_honest_nodes,
                         sybil_nodes=train_sybil_nodes + test_sybil_nodes,
                         name=name,
                         reciprocal=reciprocal)
        self.train_honest_nodes = train_honest_nodes
        self.train_sybil_nodes = train_sybil_nodes
        self.test_honest_nodes = test_honest_nodes
        self.test_sybil_nodes = test_sybil_nodes

    def train_test_split(self, train_fraction: float = None,
                         train_fraction_honest: float = 0.1,
                         train_fraction_sybil: float = 0.1):
        # Ignore arguments
        return self.train_honest_nodes, self.train_sybil_nodes, self.test_honest_nodes, self.test_sybil_nodes


class SocialNetworkFromTrainTestFiles(SocialNetworkFromTrainTestSets):
    def __init__(self,
                 network_file_name: str,
                 train_file_name: str,
                 test_file_name: str,
                 is_directed: bool,
                 network_file_type: str = "edge_list",
                 name: str = "Network",
                 reciprocal: bool = False):
        super().__init__(
            network=Graph.get_graph_from_file(network_file_name, network_file_type, is_network=True,
                                              is_directed=is_directed),
            train_honest_nodes=Graph.get_node_list_line_from_file(train_file_name, 0),
            train_sybil_nodes=Graph.get_node_list_line_from_file(train_file_name, 1),
            test_honest_nodes=Graph.get_node_list_line_from_file(test_file_name, 0),
            test_sybil_nodes=Graph.get_node_list_line_from_file(test_file_name, 1),
            name=name,
            reciprocal=reciprocal)


class SocialNetworkFromDirectory(SocialNetworkFromTrainTestFiles):
    def __init__(self,
                 directory: str,
                 is_directed: bool,
                 network_file_type: str = "edge_list",
                 name: str = "Network",
                 reciprocal: bool = False):
        super().__init__(network_file_name=f"{directory}/graph.txt",
                         train_file_name=f"{directory}/train.txt",
                         test_file_name=f"{directory}/test.txt",
                         is_directed=is_directed,
                         network_file_type=network_file_type,
                         name=name,
                         reciprocal=reciprocal)


class Twitter270K(SocialNetworkFromDirectory):
    def __init__(self,
                 is_directed: bool = True,
                 reciprocal: bool = False):
        super().__init__(directory="data/twitter-270k",
                         is_directed=is_directed,
                         network_file_type="edge_list",
                         name="Twitter270K",
                         reciprocal=reciprocal)


class Twitter270KSubsampled(SocialNetworkFromDirectory):
    def __init__(self):
        super().__init__(directory="data/twitter-270k/subsampled",
                         is_directed=False,
                         network_file_type="edge_list",
                         name="Twitter270K_subsampled")


class Twitter270KRemaining(SocialNetworkFromDirectory):
    def __init__(self):
        super().__init__(directory="data/twitter-270k/remaining",
                         is_directed=False,
                         network_file_type="edge_list",
                         name="Twitter270K_remaining")
