import networkx as nx
import matplotlib.pyplot as plt

import random

import numpy as np
import pandas as pd
import seaborn as sns


class GraphBase:
    def __init__(self, is_sybil: bool = False, is_network: bool = False,
                 name: str = "Graph") -> None:
        self.graph = nx.Graph()

        if is_sybil and is_network:
            raise Exception("The graph is both a network and sybil")

        self.is_sybil = is_sybil
        self.is_network = is_network

        self.name = name

    def __str__(self):
        graph_type = "SocialNetwork" if self.is_network else "Graph"
        return graph_type + " '" + self.name + "' " + ("[sybil] " if self.is_sybil else "") + ": " + str(self.graph)

    def set_graph(self, graph: nx.Graph):
        self.graph = graph

    def is_directed(self):
        return isinstance(self.graph, nx.DiGraph)

    def is_undirected(self):
        return not self.is_directed()

    def to_directed(self):
        if self.is_undirected():
            self.graph = self.graph.to_directed()
            return True
        return False

    def to_undirected(self, reciprocal: bool = False):
        if self.is_directed():
            # TODO determine semantics of what it means for a directed graph to be transformed to a undirected graph
            self.graph = self.graph.to_undirected(reciprocal=reciprocal)
            return True
        return False

    def undirected_graph_copy(self, reciprocal: bool = False):
        if self.is_undirected():
            return Graph(graph=self.graph.to_undirected(),
                         is_sybil=self.is_sybil,
                         is_network=self.is_network,
                         name=self.name)
        else:
            return Graph(graph=self.graph.to_undirected(reciprocal=reciprocal),
                         is_sybil=self.is_sybil,
                         is_network=self.is_network,
                         name=self.name)

    def directed_graph_copy(self):
        return Graph(graph=self.graph.to_directed(),
                     is_sybil=self.is_sybil,
                     is_network=self.is_network,
                     name=self.name)

    def nodes_list(self) -> list[int]:
        return list(self.graph.nodes())

    def num_nodes(self) -> int:
        return self.graph.order()

    def edges_list(self) -> list[(int, int)]:
        return list(self.graph.edges())

    def has_edge(self, u, v) -> bool:
        return (u, v) in self.edges_list()

    def num_edges(self) -> int:
        return len(self.edges_list())

    def adjacency_matrix(self):
        return nx.adjacency_matrix(self.graph)

    def degree(self, node):
        return self.graph.degree(node)

    def average_degree(self):
        return 2 * self.num_edges() / self.num_nodes()

    def neighbors(self, node):
        if isinstance(self.graph, nx.DiGraph):
            neighbors_out = self.neighbors_out(node)
            neighbors_in = self.neighbors_in(node)
            return list(set(neighbors_out).union(set(neighbors_in)))  # Union: all edges
        else:
            return list(self.graph.neighbors(node))

    def neighbors_bidirectional(self, node):
        if isinstance(self.graph, nx.DiGraph):
            neighbors_out = self.neighbors_out(node)
            neighbors_in = self.neighbors_in(node)
            return list(set(neighbors_out).intersection(set(neighbors_in)))  # Intersection: only bidirectional edges
        else:
            return list(self.neighbors(node))

    def neighbors_out(self, node):
        if isinstance(self.graph, nx.DiGraph):
            return list(self.graph.successors(node))
        else:
            return self.neighbors(node)

    def neighbors_in(self, node):
        if isinstance(self.graph, nx.DiGraph):
            return list(self.graph.predecessors(node))
        else:
            return self.neighbors(node)

    def single_source_shortest_path_length(self, source, cutoff=None):
        return nx.single_source_shortest_path_length(self.graph, source, cutoff=cutoff)

    def relabel_nodes(self, mapping: dict = None):
        if mapping is None:
            mapping = {}
            count = 0
            for node in self.nodes_list():
                mapping[node] = count
                count += 1

        graph_copy = self.graph.copy()
        self.graph = nx.relabel_nodes(graph_copy, mapping, copy=True)
        return mapping

    def draw(self) -> None:
        nx.draw(self.graph)

    def save_graph(self, file_name="output/graph.pdf") -> None:
        node_colors = nx.get_node_attributes(self.graph, "color")
        color_map = [node_colors[i] for i in range(len(node_colors))]
        edge_labels = nx.get_edge_attributes(self.graph, "type")

        pos = nx.spectral_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, font_size=10, font_weight='bold', node_size=1000,
                node_color=color_map)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.savefig(file_name, bbox_inches="tight")
        plt.show()

    def write_gexf(self, file_name="output/graph.gexf") -> None:
        nx.write_gexf(self.graph, file_name)

    def write_edge_list(self, file_name="output/graph.txt", both_edges_for_undirected: bool = False) -> None:
        edge_list = self.edges_list().copy()
        print(len(edge_list))
        if not isinstance(self.graph, nx.DiGraph) and both_edges_for_undirected:
            other_direction = []
            for (u, v) in edge_list:
                if u != v:
                    other_direction.append((v, u))
            edge_list.extend(other_direction)
            edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))

        print(len(edge_list))
        with open(file_name, 'w') as file:
            for (u, v) in edge_list:
                file.write(f"{u} {v}\n")
        # nx.write_edgelist(self.graph, file_name, data=False) # Doesn't print both edges for undirected


class Graph(GraphBase):
    def __init__(self, graph: nx.Graph = None, is_sybil: bool = False, is_network: bool = False,
                 name: str = "Graph"):
        super().__init__(is_sybil=is_sybil, is_network=is_network, name=name)
        if graph is not None:
            self.set_graph(graph)

    def degree_distribution(self, directory: str = "output",
                            file_name: str = None,
                            positive_nodes: [int] = None,
                            coloring: bool = False,
                            num_bins: int = None):
        if file_name is None:
            file_name = "degree_distribution.pdf"
        if positive_nodes is None:
            positive_nodes = []
        if num_bins is None:
            num_bins = 20

        # Rank-degree distribution
        degrees = dict(self.graph.degree())
        df = pd.DataFrame(list(degrees.items()), columns=['node', 'degree'])
        if positive_nodes:
            df['node_type'] = df['node'].apply(lambda x: 'positive' if x in positive_nodes else 'negative')
        # df_positive = df[df['node_type'] == 'positive']
        # df_negative = df[df['node_type'] == 'negative']

        df_rank = pd.DataFrame(list(degrees.items()), columns=['node', 'degree'])
        df_rank['rank'] = df_rank['degree'].rank(method='first', ascending=False)
        if coloring and positive_nodes:
            df_rank['node_type'] = df_rank['node'].apply(lambda x: 'Positive' if x in positive_nodes else 'Negative')

        plt.figure(figsize=(10, 6))
        if coloring:
            sns.scatterplot(data=df_rank, x='rank', y='degree', hue='node_type',
                            palette={'Positive': 'green', 'Negative': 'red'})
        else:
            sns.scatterplot(data=df_rank, x='rank', y='degree')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Rank-Degree Distribution')
        plt.xlabel('Rank')
        plt.ylabel('Degree')
        if coloring:
            plt.legend(title='Node Type')
        plt.savefig(f"{directory}/{file_name}")
        plt.show()

        # Histogram
        # degree_sequence = [d for n, d in self.graph.degree()]

        plt.figure(figsize=(10, 6))

        # degree_counts = pd.Series(degree_sequence).value_counts().sort_index()
        # ax = sns.barplot(x=degree_counts.index, y=degree_counts.values)
        # max_degree = max(degree_counts.index)
        # max_x_tick = ((max_degree - 1) // 10 + 1) * 10

        # sns.histplot(degree_sequence_negative, ax=ax1, bins=num_bins, kde=False)
        if positive_nodes:
            sns.displot(
                df, x="degree", col="node_type", stat="percent",
                bins=50, height=3, facet_kws=dict(margin_titles=True),
            )
        else:
            sns.displot(
                df, x="degree", stat="percent",
                bins=50, height=3, facet_kws=dict(margin_titles=True),
            )

        plt.savefig(f"{directory}/hist_{file_name}")
        plt.show()

        # plt.figure(figsize=(10, 6))
        # degree_sequence_positive = [d for n, d in self.graph.degree() if n in positive_nodes]
        # degree_sequence_negative = [d for n, d in self.graph.degree() if n not in positive_nodes]
        # plt.hist([degree_sequence_negative, degree_sequence_positive],
        #          bins=10, stacked=True, stat="percent",
        #          label=['Negative', 'Positive'])
        # plt.savefig(f"{directory}/hist2_{file_name}")
        # plt.show()

        if positive_nodes:
            for log_scale in [True, False]:
                plt.figure(figsize=(10, 6))
                sns.displot(
                    df, x="degree", hue="node_type", multiple="stack", stat="percent", log_scale=log_scale,
                    bins=50, height=3, facet_kws=dict(margin_titles=True)
                )
                plt.savefig(f"{directory}/hist_col_{log_scale}_{file_name}")
                plt.show()

    def structural_analysis(self, sybil_fraction: float = 0.5, num_experiments: int = 1):
        print(f"Structural analysis of {self.__class__.__name__}")
        num_nodes = self.num_nodes()
        num_sybils = round(sybil_fraction * num_nodes)
        num_honests = num_nodes - num_sybils

        nodes_list = self.nodes_list()

        experiments = []

        for _ in range(num_experiments):
            num_attack_edges = [0, 0, 0]
            sybil_nodes = random.sample(nodes_list, k=num_sybils)
            honest_nodes = list(set(nodes_list) - set(sybil_nodes))

            labels = np.zeros(num_nodes)
            labels[sybil_nodes] = 1
            labels[honest_nodes] = -1

            for (u, v) in self.edges_list():
                if labels[u] == -1 and labels[v] == 1 or labels[u] == 1 and labels[v] == -1:
                    if not self.is_directed():
                        num_attack_edges[0] += 1  # undirected graph, just save in index 0
                    elif labels[u] == -1 and labels[v] == 1 and labels[u] == 1 and labels[v] == -1:
                        num_attack_edges[0] += 1  # directed graph, save bidirectional edge (index 0)
                    elif labels[u] == -1 and labels[v] == 1:
                        num_attack_edges[1] += 1  # directed graph, edge honest -> sybil (index 1)
                    elif labels[u] == 1 and labels[v] == -1:
                        num_attack_edges[2] += 1  # directed graph, edge sybil -> honest (index 2)
                    else:
                        raise Exception("Shouldn't be possible")
            if self.is_directed():
                experiments.append(num_attack_edges)
            else:
                experiments.append(num_attack_edges[0])

        # print(experiments)
        mean = np.mean(experiments, axis=0)
        # print(f"mean = {mean}")
        if isinstance(mean, np.ndarray):
            attack_edges = mean[0] + mean[2]
        else:
            attack_edges = mean
        attack_edge_per_sybil = attack_edges / num_sybils
        print(f"attack edges per sybil {attack_edge_per_sybil}")
        average_node_degree = self.average_degree()
        print(f"avg node degree = {average_node_degree}")
        return average_node_degree, attack_edge_per_sybil

    @staticmethod
    def write_node_lists_to_file(node_lists: [[int]], file_name: str):
        with open(file_name, 'w') as file:
            for node_list in node_lists:
                file.write(' '.join(str(x) for x in node_list))
                file.write('\n')

    @staticmethod
    def get_graph_from_file(file_name: str, file_type: str = "edge_list", is_network: bool = False,
                            is_directed: bool = False):
        if file_type == "edge_list":
            return EdgeListGraph(file_name, is_network=is_network, is_directed=is_directed)
        elif file_type == "gexf":
            return GEXFGraph(file_name, is_network=is_network)
        else:
            raise Exception("Unknown file type")

    @staticmethod
    def get_node_list_from_file(file_name: str) -> [int]:
        with open(file_name) as file:
            return [int(x) for x in file.readline().split()]

    @staticmethod
    def get_node_list_line_from_file(file_name: str, line_idx: int) -> [int]:
        with open(file_name) as file:
            lines = file.readlines()
            return [int(x) for x in lines[line_idx].split()]

    @staticmethod
    def get_node_list_with_probabilities_from_file(file_name: str) -> [int]:
        with open(file_name) as file:
            values = [int(x) for x in file.readline().split()]

        if len(values) % 2 != 0:
            raise Exception(
                "Error in file format, is supposed to be n nodes, where each node is 'n_i p_i', n_i being the node_id "
                "and p_i being the probability of being a sybil node")

        num_nodes = len(values) // 2
        return [(values[2 * i], values[2 * i + 1]) for i in range(num_nodes)]


class GEXFGraph(Graph):
    def __init__(self, file_name: str,
                 node_type=int,
                 is_sybil: bool = False,
                 is_network: bool = False) -> None:
        super().__init__(is_sybil=is_sybil, is_network=is_network)
        self.file_name = file_name
        self._import_graph(node_type=node_type)

    def _import_graph(self, node_type) -> None:
        graph = nx.read_gexf(self.file_name, node_type=node_type)

        if not isinstance(graph, nx.Graph):
            raise Exception("Import failed")

        self.set_graph(graph)


class EdgeListGraph(Graph):
    def __init__(self, file_name: str,
                 is_sybil: bool = False,
                 is_network: bool = False,
                 is_directed: bool = False) -> None:
        super().__init__(is_sybil=is_sybil, is_network=is_network)
        self.file_name = file_name
        self._import_graph(is_directed)

    def _import_graph(self, is_directed: bool) -> None:
        if is_directed:
            graph = nx.read_edgelist(path=self.file_name, nodetype=int, create_using=nx.DiGraph)
        else:
            graph = nx.read_edgelist(path=self.file_name, nodetype=int)

        if not isinstance(graph, nx.Graph):
            raise Exception("Import failed")

        self.set_graph(graph)


class FacebookSNAP(EdgeListGraph):
    def __init__(self, is_sybil: bool = False):
        super().__init__(file_name="data/snap-facebook/graph.txt",
                         is_sybil=is_sybil,
                         is_network=False,
                         is_directed=False)
        self.name = "snap_facebook"


class BarabasiAlbertGraph(Graph):
    def __init__(self, n, m, seed=None, initial_graph=None, is_sybil: bool = False,
                 is_network: bool = False) -> None:
        super().__init__(is_sybil=is_sybil, is_network=is_network, name=f"BarabasiAlbert(m={m})")
        graph = nx.barabasi_albert_graph(n, m, seed, initial_graph)
        self.set_graph(graph)


class DualBarabasiAlbertGraph(Graph):
    def __init__(self, n, m1, m2, p, seed=None, initial_graph=None, is_sybil: bool = False,
                 is_network: bool = False) -> None:
        super().__init__(is_sybil=is_sybil, is_network=is_network, name="DualBarabasiAlbert")
        graph = nx.dual_barabasi_albert_graph(n, m1, m2, p, seed, initial_graph)
        self.set_graph(graph)


class ExtendedBarabasiAlbertGraph(Graph):
    def __init__(self, n, m, p: float = 0.0, q: float = 0.0, seed=None,
                 is_sybil: bool = False,
                 is_network: bool = False) -> None:
        super().__init__(is_sybil=is_sybil, is_network=is_network, name="ExtendedBarabasiAlbert")
        graph = nx.extended_barabasi_albert_graph(n, m, p, q, seed)
        self.set_graph(graph)


class PowerLawGraph(Graph):
    def __init__(self, n, m, p, seed=None, is_sybil: bool = False, is_network: bool = False) -> None:
        super().__init__(is_sybil=is_sybil, is_network=is_network, name=f"PowerLaw(m={m}, p={p})")
        graph = nx.powerlaw_cluster_graph(n, m, p, seed)
        self.set_graph(graph)


class PreferentialAttachmentGraph(Graph):
    def __init__(self, aseq, p, create_using=None, seed=None, is_sybil: bool = False, is_network: bool = False) -> None:
        super().__init__(is_sybil=is_sybil, is_network=is_network, name="PreferentialAttachment")
        graph = nx.bipartite.preferential_attachment_graph(aseq, p, create_using, seed)
        self.set_graph(graph)


class SmallWorldGraph(Graph):  # Kleinberg
    def __init__(self, n: int, p: int, q: int, r: float, dim: int = 2, seed=None, is_sybil: bool = False,
                 is_network: bool = False) -> None:
        super().__init__(is_sybil=is_sybil, is_network=is_network, name="SmallWorld")
        graph = nx.navigable_small_world_graph(n, p, q, r, dim, seed)
        self.set_graph(graph)

    def nodes_list(self, convert_to_node_ids: bool = True) -> list[int]:
        nodes_list = super().nodes_list()
        if not convert_to_node_ids:
            return nodes_list
        max_idx = max([max(x1, x2) for (x1, x2) in nodes_list])
        nodes_list = [(n1 * max_idx + n2) for (n1, n2) in nodes_list]
        return nodes_list

    def edges_list(self, convert_to_node_ids: bool = True) -> list[(int, int)]:
        edges_list = super().edges_list()
        if not convert_to_node_ids:
            return edges_list
        max_idx = max([max([x1, x2, x3, x4]) for ((x1, x2), (x3, x4)) in edges_list])
        edges_list = [((n1 * max_idx + n2), (n3 * max_idx + n4)) for ((n1, n2), (n3, n4)) in edges_list]
        return edges_list


class ErdosRenyiGraph(Graph):
    def __init__(self, n: int, p: float, seed=None, directed: bool = False, is_sybil: bool = False,
                 is_network: bool = False) -> None:
        super().__init__(is_sybil=is_sybil, is_network=is_network, name=f"ErdosRenyi(p={p})")
        graph = nx.erdos_renyi_graph(n, p, seed, directed)
        self.set_graph(graph)


class StochasticBlockModelGraph():
    def __init__(self, sizes: [int], p: [[float]], seed=None, directed: bool = False, selfloops: bool = False,
                 sparse: bool = True, is_sybil: bool = False, is_network: bool = False):
        super().__init__(is_sybil=is_sybil, is_network=is_network, name="StochasticBlockModel")
        graph = nx.stochastic_block_model(sizes=sizes, p=p, seed=seed, directed=directed, selfloops=selfloops,
                                          sparse=sparse)
        self.set_graph(graph)
