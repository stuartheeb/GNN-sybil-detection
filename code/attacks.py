import random
import numpy as np


class Attack:
    def __init__(self,
                 pdf_targeted: [float],
                 p_targeted: float = 1.0,
                 use_all_honest_nodes: bool = False,
                 use_all_sybil_nodes: bool = True,
                 name: str = "TargetedAttack",
                 seed=None) -> None:
        self.pdf_targeted = pdf_targeted
        self.p_targeted = p_targeted

        self.use_all_honest_nodes = use_all_honest_nodes
        self.use_all_sybil_nodes = use_all_sybil_nodes

        self.social_network = None
        self.honest_targets = None
        self.sybil_targets = None
        self.num_attack_edges = None

        suffix = f"(p_t={self.p_targeted}, pdf={self.pdf_targeted})" if self.p_targeted != 0.0 else ""
        self.name = name + suffix

        self.seed = seed

    def __str__(self):
        return self.name

    def _verify(self):
        if not np.isclose(sum(self.pdf_targeted), 1.0):
            raise Exception("The sum of the probabilities must be 1.0")

        if self.num_attack_edges is None:
            raise Exception("Number of attack edges (<->, ->, <-) must be provided")

        if len(self.num_attack_edges) != 3:
            raise Exception("Number of attack edges must be provided for each type of edge (<->, ->, <-)")

        if not all(isinstance(x, int) for x in self.num_attack_edges):
            raise Exception("Number of attack edges must be integers")

        if min(self.num_attack_edges) < 0:
            raise Exception("Number of attack edges must be non-negative")

        if not self.social_network.network.is_directed():
            if self.num_attack_edges[1] is None or self.num_attack_edges[1] != 0:
                raise Exception("Number of attack edges -> was not zero but the given graph is undirected")
            if self.num_attack_edges[2] is None or self.num_attack_edges[2] != 0:
                raise Exception("Number of attack edges <- was not zero but the given graph is undirected")

        if self.social_network.honest_nodes is None or len(self.social_network.honest_nodes) == 0 \
                or self.social_network.sybil_nodes is None or len(self.social_network.sybil_nodes) == 0:
            raise Exception("Honest and sybil nodes must be provided")

        if self.honest_targets is None or len(self.honest_targets) == 0 or self.sybil_targets is None or len(
                self.sybil_targets) == 0:
            raise Exception("Honest and sybil targets must be provided")

        num_honest_nodes, num_sybil_nodes = self._get_number_of_nodes()
        max_attack_edges = num_honest_nodes * num_sybil_nodes
        if sum(self.num_attack_edges) > max_attack_edges:
            raise Exception("The number of attack edges is too high")

    def _get_number_of_nodes(self):
        return len(self.honest_targets), len(self.sybil_targets)

    def perform_attack(self, social_network,
                       num_attack_edges: [int],
                       honest_targets: [int] = None,
                       sybil_targets: [int] = None):

        self.social_network = social_network

        if honest_targets is None or self.use_all_honest_nodes:
            self.honest_targets = social_network.honest_nodes
        else:
            self.honest_targets = honest_targets

        if sybil_targets is None or self.use_all_sybil_nodes:
            self.sybil_targets = social_network.sybil_nodes
        else:
            self.sybil_targets = sybil_targets

        if not social_network.network.is_directed() and not isinstance(num_attack_edges, list):
            num_attack_edges = [num_attack_edges, 0, 0]

        self.num_attack_edges = num_attack_edges
        self._verify()

        attack_edges = self._get_attack_edges()

        social_network.add_edges(attack_edges)

    def _get_attack_edges(self):
        edges_to_add = []
        for i in range(len(self.num_attack_edges)):
            edges_to_add_i = []
            while len(edges_to_add_i) != self.num_attack_edges[i]:
                targeted_edge = False
                p = random.uniform(0, 1)
                if (p <= self.p_targeted or np.isclose(self.p_targeted, 1.0)) and not np.isclose(self.p_targeted, 0.0):
                    targeted_edge = True
                    sources = self.honest_targets
                    targets = self.sybil_targets
                else:
                    sources = self.social_network.honest_nodes
                    targets = self.social_network.sybil_nodes

                if i == 2:  # <- attack edge, swap
                    sources, targets = targets, sources

                success = False
                while not success:
                    u = random.choice(sources)
                    v = random.choice(targets)
                    if targeted_edge:
                        p = random.uniform(0, 1)
                        j = 0
                        while p > self.pdf_targeted[j] and j < len(self.pdf_targeted) - 1:
                            p -= self.pdf_targeted[j]
                            j += 1

                        k_hop_distance = j
                        if k_hop_distance > 0:
                            node_distances = self.social_network.network.single_source_shortest_path_length(v,
                                                                                                            cutoff=k_hop_distance)
                            k_hop_nodes = [node for node, length in node_distances.items() if length == k_hop_distance]
                            if len(k_hop_nodes) == 0:
                                continue
                            v = random.choice(k_hop_nodes)

                    edge = (u, v)
                    if edge in edges_to_add_i:
                        continue
                    edges_to_add_i.append(edge)
                    success = True
                    if i == 0 and self.social_network.network.is_directed():
                        # <-> attack edge, case of i == 2 will be handled symmetrically with the code above
                        edge = (v, u)
                        edges_to_add_i.append(edge)

            edges_to_add.extend(edges_to_add_i)
        return edges_to_add


class RandomAttack(Attack):
    def __init__(self, name: str = "RandomAttack", seed=None) -> None:
        super().__init__(pdf_targeted=[1.0],
                         p_targeted=0.0,
                         use_all_honest_nodes=True,
                         use_all_sybil_nodes=True,
                         name=name,
                         seed=seed)
