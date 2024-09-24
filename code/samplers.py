import copy

from social_networks import *
import math
import littleballoffur as lbof


class Sampler:
    def __init__(self):
        pass

    def sample_graph(self, graph: Graph, sources: [int], number_of_nodes: int,
                     allow_more_nodes_to_round: bool = False,
                     verbose: bool = True) -> Graph:
        raise Exception("This is the base sampler class, it has no functionality")

    def sample_social_network(self,
                              social_network: SocialNetwork,
                              number_of_nodes: int,
                              num_sources: int,
                              source_split: str = "equal",
                              allow_more_nodes_to_round: bool = False,
                              verbose: bool = True) -> (SocialNetwork, SocialNetwork):
        original_social_network = copy.deepcopy(social_network)

        if source_split == "equal":
            k_h = math.ceil(num_sources / 2)
            k_s = num_sources - k_h
        elif source_split == "proportional":
            k_h = math.ceil(
                num_sources * len(original_social_network.honest_nodes) / original_social_network.network.num_nodes())
            k_s = num_sources - k_h
        else:
            raise Exception("Unknown source split")

        sources_honest = random.sample(original_social_network.honest_nodes, k_h)
        sources_sybil = random.sample(original_social_network.sybil_nodes, k_s)

        subsampled_network_graph = self.sample_graph(graph=original_social_network.network,
                                                     sources=sources_honest + sources_sybil,
                                                     number_of_nodes=number_of_nodes,
                                                     verbose=verbose)
        if verbose:
            print(f"Actual sampling process complete")
        sampled_nodes = subsampled_network_graph.nodes_list()
        new_train_honest_nodes1 = set(sampled_nodes).intersection(original_social_network.train_honest_nodes)
        new_train_sybil_nodes1 = set(sampled_nodes).intersection(original_social_network.train_sybil_nodes)
        new_test_honest_nodes1 = set(sampled_nodes).intersection(original_social_network.test_honest_nodes)
        new_test_sybil_nodes1 = set(sampled_nodes).intersection(original_social_network.test_sybil_nodes)
        subsampled_social_network = SocialNetworkFromTrainTestSets(network=subsampled_network_graph,
                                                                   train_honest_nodes=list(new_train_honest_nodes1),
                                                                   train_sybil_nodes=list(new_train_sybil_nodes1),
                                                                   test_honest_nodes=list(new_test_honest_nodes1),
                                                                   test_sybil_nodes=list(new_test_sybil_nodes1))
        if verbose:
            print(f"Sampled social network created")

        non_sampled_nodes = list(
            set(original_social_network.network.nodes_list()) - set(subsampled_social_network.network.nodes_list()))
        remaining_network_graph = Graph(graph=original_social_network.network.graph.subgraph(non_sampled_nodes))
        new_train_honest_nodes2 = set(non_sampled_nodes).intersection(original_social_network.train_honest_nodes)
        new_train_sybil_nodes2 = set(non_sampled_nodes).intersection(original_social_network.train_sybil_nodes)
        new_test_honest_nodes2 = set(non_sampled_nodes).intersection(original_social_network.test_honest_nodes)
        new_test_sybil_nodes2 = set(non_sampled_nodes).intersection(original_social_network.test_sybil_nodes)
        remaining_social_network = SocialNetworkFromTrainTestSets(
            network=remaining_network_graph,
            train_honest_nodes=list(new_train_honest_nodes2),
            train_sybil_nodes=list(new_train_sybil_nodes2),
            test_honest_nodes=list(new_test_honest_nodes2),
            test_sybil_nodes=list(new_test_sybil_nodes2))
        if verbose:
            print(f"Remaining social network created")

        subsampled_social_network.relabel_network()
        if verbose:
            print(f"Subsampled social network relabeled")
        remaining_social_network.relabel_network()
        if verbose:
            print(f"Remaining social network relabeled")

        return subsampled_social_network, remaining_social_network


class BFSSampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample_graph(self, graph: Graph, sources: [int], number_of_nodes: int,
                     allow_more_nodes_to_round: bool = False,
                     verbose: bool = True) -> Graph:
        return Graph()


class LBOFSampler(Sampler):
    def __init__(self, lbof_sampler):
        super().__init__()

        self.lbof_sampler = lbof_sampler

    def sample_graph(self,
                     graph: Graph,
                     sources: [int],
                     number_of_nodes: int,
                     allow_more_nodes_to_round: bool = False,
                     verbose: bool = True) -> Graph:
        if verbose:
            print(f"Sampling {graph}")
        internal_graph = graph.graph
        num_sources = len(sources)

        sampled_nodes = set()
        if isinstance(self.lbof_sampler, lbof.ForestFireSampler):
            self.lbof_sampler.number_of_nodes = number_of_nodes
            sampled = self.lbof_sampler.sample(graph=internal_graph)
            # print(sampled.nodes)
            sampled_nodes = sampled_nodes.union(sampled.nodes)
        else:
            if verbose:
                print(f"sources (n={len(sources)}) = {sources}")
            self.lbof_sampler.number_of_nodes = math.ceil(number_of_nodes / num_sources)
            for source in sources:
                if isinstance(self.lbof_sampler, lbof.MetropolisHastingsRandomWalkSampler):
                    sampled = self.lbof_sampler.sample(graph=internal_graph, start_node=source)
                else:
                    raise Exception("Unknown sampler")
                # print(sampled.nodes)
                sampled_nodes = sampled_nodes.union(sampled.nodes)
        # print(f"sampled_nodes = {sampled_nodes}")

        return Graph(graph=internal_graph.subgraph(sampled_nodes))


class ForestFireSampler(LBOFSampler):
    def __init__(self, p: float = 0.4, seed=None):
        super().__init__(lbof_sampler=lbof.ForestFireSampler(p=p, seed=seed))


class MetropolisHastingsRandomWalkSampler(LBOFSampler):
    def __init__(self, seed=None):
        super().__init__(lbof_sampler=lbof.MetropolisHastingsRandomWalkSampler(seed=seed))
