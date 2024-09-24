from graphs import *
import math
import operator
import numpy as np
import utils
import subprocess
from scipy.sparse import *
import os
import time


class SybilFinder:
    def __init__(self,
                 graph: Graph,
                 honest_nodes: [int],
                 sybil_nodes: [int],
                 uses_directed_graph: bool,
                 uses_honest_nodes: bool,
                 uses_sybil_nodes: bool,
                 has_trust_values: bool) -> None:

        self.uses_directed_graph = uses_directed_graph

        self.uses_honest_nodes = uses_honest_nodes
        self.uses_sybil_nodes = uses_sybil_nodes

        self.has_trust_values = has_trust_values
        self.trust_values = None

        # Set graph
        self.graph = None
        self.nodes_list = None
        if graph is not None:
            self.set_graph(graph)

        # Set train nodes
        self.train_labels = None
        self._honest_nodes = None
        self._sybil_nodes = None
        self.num_honest_nodes = 0
        self.num_sybil_nodes = 0
        if honest_nodes is not None and sybil_nodes is not None:
            self.set_train_nodes(honest_nodes=honest_nodes, sybil_nodes=sybil_nodes)

        self.predicted_sybils = None
        self.pretrain_runtime= None
        self.runtime = None

        self.name = "SybilFinder(Base)"

        self.verbose = True

    def __str__(self):
        return self.name

    def _description(self):
        return "Base class, no functionality"

    def set_graph(self, graph: Graph) -> None:
        self.runtime = None
        if self.verbose:
            print(f"Setting graph: {graph}")
        if self.uses_directed_graph:
            if graph.to_directed():
                if self.verbose:
                    print("Graph transformed to directed ...")
        elif not self.uses_directed_graph:
            if graph.to_undirected():
                if self.verbose:
                    print("Graph transformed to undirected ...")

        self.graph = graph
        self.nodes_list = self.graph.nodes_list()
        self.train_labels = np.zeros(self.graph.num_nodes())
        self._verify_graph()

    def _verify_graph(self):
        # Check if IDs are 0,...,num_nodes-1
        verified = True
        nodes = self.graph.nodes_list()
        num_nodes = self.graph.num_nodes()
        nodes_set = set(nodes)
        verified = verified and len(nodes_set) == num_nodes
        verified = verified and min(nodes) == 0 and max(nodes) == num_nodes - 1
        if not verified:
            raise Exception(f"Graph verification failed min(nodes): {min(nodes)}, max(nodes): {max(nodes)}")

    def set_train_nodes(self, honest_nodes: [int] = None, sybil_nodes: [int] = None) -> None:
        if self.uses_honest_nodes:
            self._honest_nodes = honest_nodes
            for i in honest_nodes:
                self.train_labels[i] = -1
            self.num_honest_nodes = len(honest_nodes)
        if self.uses_sybil_nodes:
            self._sybil_nodes = sybil_nodes
            for i in sybil_nodes:
                self.train_labels[i] = 1
            self.num_sybil_nodes = len(sybil_nodes)

    def find_sybils(self) -> list[int]:
        if self.verbose:
            print(f"\nRunning {self.__str__()} ...")
            print(f"Graph: {self.graph}")
            print(self._description())
            unique, counts = np.unique(self.train_labels, return_counts=True)
            print(f"Train labels: {dict(zip(unique, counts))}")

        start_time = time.time()
        predicted_sybils = self._find_sybils()
        end_time = time.time()
        self.runtime = round((end_time - start_time) * 1000)  # in ms

        return predicted_sybils

    def _find_sybils(self) -> list[int]:
        raise NotImplementedError("This method must be implemented by the subclass")

    def sybil_classification(self, values, threshold: float, flip: bool = False) -> list[int]:

        if flip:
            values = -values
            threshold = -threshold

        nodes_list = self.nodes_list  # self.graph.nodes_list()

        predicted_sybils = []
        for i in range(len(values)):
            if values[i] > threshold:
                # TODO: RETHINK !
                predicted_sybils.append(i)
                # predicted_sybils.append(nodes_list[i])
        return predicted_sybils


class SybilFinderRandom(SybilFinder):

    def __init__(self, graph: Graph = None, p: float = 0.5) -> None:
        super().__init__(graph=graph,
                         honest_nodes=None,
                         sybil_nodes=None,
                         uses_directed_graph=False,
                         uses_honest_nodes=False,
                         uses_sybil_nodes=False,
                         has_trust_values=True)

        self.p = p

        self.name = "SybilFinderRandom"

    def _find_sybils(self) -> list[int]:
        n = self.graph.num_nodes()
        values = np.random.uniform(0, 1, n)

        self.trust_values = np.random.uniform(0, 1, n)

        return self.sybil_classification(values, self.p)


class LegacyAlgorithm(SybilFinder):
    def __init__(self,
                 algorithm_name: str,
                 graph: Graph = None,
                 honest_nodes: list[int] = None,
                 sybil_nodes: list[int] = None,
                 max_iter: int = 20,
                 threshold: float = 0.5
                 ):
        super().__init__(graph=graph,
                         honest_nodes=honest_nodes,
                         sybil_nodes=sybil_nodes,
                         uses_directed_graph=True if algorithm_name == "GANG" else False,
                         uses_honest_nodes=True,
                         uses_sybil_nodes=False if algorithm_name == "SybilRank" else True,
                         has_trust_values=True)
        self.algorithm_name = algorithm_name
        self.max_iter = max_iter
        self.threshold = threshold

        self.directory = "data/legacy_temp"
        self.graph_file = f"{self.directory}/graph.txt"
        self.train_file = f"{self.directory}/train.txt"
        self.test_file = f"{self.directory}/test.txt"
        self.post_file = f"{self.directory}/post_{algorithm_name}.txt"

        self.name = algorithm_name + "(Legacy)"

    def retrieve_post_values(self):
        if not os.path.exists(self.post_file):
            raise Exception("Post file does not exist.")
        array = np.loadtxt(self.post_file)
        post_values = array[:, 1]
        return post_values

    def get_process_list(self):
        process_list = None
        if self.algorithm_name == "SybilRank":
            process_list = ["baselines_code/sybilrank",
                            "-graphfile", self.graph_file,
                            "-trainfile", self.train_file,
                            "-postfile", self.post_file,
                            "-alpha", "0",
                            "-mIter", str(self.max_iter)]
        elif self.algorithm_name == "SybilBelief":
            process_list = ["baselines_code/sybilbelief",
                            "-graphfile", self.graph_file,
                            "-trainfile", self.train_file,
                            "-postfile", self.post_file,
                            "-mIter", str(self.max_iter),
                            "-nt", "1"]
        elif self.algorithm_name == "SybilSCAR":
            process_list = ["baselines_code/sybilscar",
                            "-graphfile", self.graph_file,
                            "-trainfile", self.train_file,
                            "-postfile", self.post_file,
                            "-mIter", str(self.max_iter),
                            "-nt", "1"]
        elif self.algorithm_name == "GANG":
            process_list = ["baselines_code/GANG",
                            "-graphfile", self.graph_file,
                            "-trainfile", self.train_file,
                            "-postfile", self.post_file,
                            "-mIter", str(self.max_iter),
                            "-nt", "1"]
        return process_list

    def _find_sybils(self) -> list[int]:

        process_list = self.get_process_list()

        if process_list is not None:
            for i in range(10):
                if not os.path.exists(self.post_file):
                    print(f"Waited for {i} seconds until file was definitely not here")
                    break
                time.sleep(1)

            if os.path.exists(self.post_file):
                print("Warning: Post file from last time is still here.")

            run_algo = subprocess.Popen(process_list, stdout=subprocess.PIPE)
            for i in range(20):
                if os.path.exists(self.post_file):
                    print(f"Waited for {i} seconds until file appeared")
                    break
                time.sleep(1)
            if not os.path.exists(self.post_file):
                print(f"FILE DIDN'T APPEAR")
            # os.listdir(self.directory)
            # time.sleep(5)
            post = self.retrieve_post_values()
            self.trust_values = post
            if post.shape[0] != self.graph.num_nodes():
                print(
                    f"Legacy algorithm failed to generate correct post values, predicting no Sybils. Post has size {post.shape[0]}, but expected values for {self.graph.num_nodes()} nodes.")
                return []

            os.remove(self.post_file)
            for i in range(10):
                if not os.path.exists(self.post_file):
                    print(f"Waited for {i} seconds until file disappeared")
                    break
                time.sleep(1)

            if os.path.exists(self.post_file):
                print("Warning: Post file not guaranteed to be removed.")

            return self.sybil_classification(post, self.threshold)
        else:
            raise Exception("Process list could not be generated.")


class SybilRank(SybilFinder):
    def __init__(self,
                 graph: Graph = None,
                 honest_nodes: [int] = None,
                 total_trust: float = 1e3,
                 pivot: float = 0.5,
                 num_iterations_multiplier: float = 1.0):
        super().__init__(graph=graph,
                         honest_nodes=honest_nodes,
                         sybil_nodes=None,
                         uses_directed_graph=False,
                         uses_honest_nodes=True,
                         uses_sybil_nodes=False,
                         has_trust_values=True)

        self.total_trust = total_trust
        self.pivot = pivot
        self.num_iterations_multiplier = num_iterations_multiplier
        self.max_iter = math.ceil(math.log2(270000))

        self.name = "SybilRank"

    def _description(self):
        return f"Number of iterations: {self.max_iter}"

    def set_graph(self, graph: Graph) -> None:
        super().set_graph(graph)
        self.max_iter = int(math.ceil(math.log2(self.graph.num_nodes())) * self.num_iterations_multiplier)

    def _find_sybils(self) -> list[int]:
        if self.max_iter is None:
            raise Exception("Max iterations not set")

        trust = self._initialize_trust()

        for i in range(self.max_iter):
            trust = self._propagate_trust(trust)

        normalized_trust = self.degree_normalize_trust(trust, self.graph)

        self.trust_values = normalized_trust

        # Ranked trust will be dict
        ranked_trust = self.rank_trust(normalized_trust)

        default_pivot_idx = int(self.pivot * len(ranked_trust))

        return [node for node, trust in ranked_trust[:default_pivot_idx]]

    def _initialize_trust(self):

        trust = np.zeros(self.graph.num_nodes())

        honest_trust = self.total_trust / self.num_honest_nodes
        # honest_trust = 1.0  # TODO reevaluate

        # These are the seed honest nodes
        for i in self.graph.nodes_list():
            if self.train_labels[i] == -1:
                trust[i] = honest_trust

        return trust

    def _propagate_trust(self, current_trust):
        # updated_trust = {node: 0 for node in self.graph.nodes_list()}
        updated_trust = np.zeros(self.graph.num_nodes())
        # for node, t in current_trust.items():
        for node in range(current_trust.shape[0]):

            # OPTION 1 -- Update node trust based on neighbors and neighbor's trust and degree
            # new_trust = 0.0
            # neighbors = self.graph.neighbors(node)
            #
            # for neighbor in neighbors:
            #     neighbor_degree = self.graph.degree(neighbor)
            #     new_trust += current_trust[neighbor] / neighbor_degree
            #
            # updated_trust[node] = new_trust

            # OPTION 2 -- Update neighbor trust base on node trust and degree
            # This option seems to be more efficient
            neighbors = self.graph.neighbors(node)
            node_degree = self.graph.degree(node)
            for neighbor in neighbors:
                updated_trust[neighbor] += current_trust[node] / node_degree

        return updated_trust

    @staticmethod
    def degree_normalize_trust(trust, graph):
        normalized_trust = trust.copy()
        for node in range(trust.shape[0]):
            node_degree = graph.degree(node)
            if node_degree > 0:
                normalized_trust[node] = trust[node] / node_degree

        return normalized_trust

    @staticmethod
    def rank_trust(trust):
        trust = utils.np_array_to_dict(trust)
        ranked_trust = sorted(
            iter(trust.items()),
            key=operator.itemgetter(1)
        )
        return ranked_trust


class SybilBelief(SybilFinder):
    def __init__(self,
                 graph: Graph = None,
                 honest_nodes: list[int] = None,
                 sybil_nodes: list[int] = None,  # Optional
                 max_iter: int = 20,
                 threshold: float = 0.5,
                 h: {} = None,
                 J: float = 0.9,
                 epsilon: float = 1e-3) -> None:
        super().__init__(graph=graph,
                         honest_nodes=honest_nodes,
                         sybil_nodes=sybil_nodes,
                         uses_directed_graph=False,
                         uses_honest_nodes=True,
                         uses_sybil_nodes=True,  # Optional
                         has_trust_values=True)

        self.max_iter = max_iter
        self.threshold = threshold

        self.h = h
        self.J = J

        self.epsilon = epsilon

        self.name = "SybilBelief"

    def _description(self):
        return f"iterations = {self.max_iter}"

    def _find_sybils(self) -> list[int]:

        if self.h is None or True:  # TODO check again
            self.h = {node: 0 for node in range(self.graph.num_nodes())}
            # h = +- 2 seem to be empirically good values
            for i in range(self.graph.num_nodes()):
                if self.train_labels[i] == -1:
                    self.h[i] = 2.0
                if self.train_labels[i] == 1:
                    self.h[i] = -2.0

        beliefs = self._initialize_beliefs()

        for i in range(self.max_iter):
            new_beliefs = np.zeros_like(beliefs)
            for node in self.graph.nodes_list():
                total_influence = sum(self.J * (2 * beliefs[neighbor] - 1) for neighbor in self.graph.neighbors(node))
                new_beliefs[node] = 1 / (1 + np.exp(-self.h[node] - total_influence))

            # if max(abs(b_t[node] - b_t_1[node]) for node in self.graph.nodes_list()) < self.epsilon:
            if np.linalg.norm(new_beliefs - beliefs, ord=1) < self.epsilon:
                if self.verbose:
                    print(f"Terminating after {i + 1} iterations")
                break

            beliefs = new_beliefs.copy()

        self.trust_values = beliefs  # b_t is belief that node is honest
        self.has_trust_values = True

        return self.sybil_classification(1 - beliefs, self.threshold)

    def _initialize_beliefs(self):
        beliefs = np.full(self.graph.num_nodes(), 0.5)
        for i in range(self.graph.num_nodes()):
            if self.train_labels[i] == -1:
                beliefs[i] = 0.9  # TODO was 1.0 before
            if self.train_labels[i] == 1:
                beliefs[i] = 0.1  # TODO was 0.0 before

        return beliefs


class SybilBeliefNew():
    def __init__(self,
                 graph: Graph = None,
                 honest_nodes: list[int] = None,
                 sybil_nodes: list[int] = None,  # Optional
                 max_iter: int = 20,
                 w: float = 0.9,
                 threshold: float = 0.5,
                 epsilon: float = 1e-3) -> None:
        pass


class SybilSCAR(SybilFinder):
    def __init__(self,
                 graph: Graph = None,
                 honest_nodes: list[int] = None,
                 sybil_nodes: list[int] = None,
                 variant: str = "D",
                 max_iter: int = 20,
                 threshold: float = 0.5,
                 const_homophily_strength: float = None,  # if not specified will be set to 1 / (2*avg_degree)
                 theta: float = 0.1,
                 delta: float = 1e-3
                 ):
        super().__init__(graph=graph,
                         honest_nodes=honest_nodes,
                         sybil_nodes=sybil_nodes,
                         uses_directed_graph=False,
                         uses_honest_nodes=True,
                         uses_sybil_nodes=True,
                         has_trust_values=True)

        if variant not in ["C", "D"]:
            raise Exception("SybilSCAR variant must be either 'C' or 'D', variant", variant, "is not supported")

        self.variant = variant
        self.max_iter = max_iter
        self.threshold = threshold
        self.const_homophily_strength = const_homophily_strength
        self.theta = theta
        self.delta = delta

        self.name = "SybilSCAR"
        if variant == "C":
            self.name += "-C"

    def _description(self):
        return f"Number of iterations: {self.max_iter}"

    def _find_sybils(self) -> list[int]:
        if self.variant == "C" and self.const_homophily_strength is None:
            # Default value suggested by paper (depends on graph)
            self.const_homophily_strength = 1.0 / (2.0 * self.graph.average_degree())  # TODO Check again in paper

        priors = self._initialize_priors()

        A = self.graph.adjacency_matrix()
        # Homophily strength matrix
        W = self._homophily_strength_matrix(A)

        p = priors - 0.5  # shifted priors
        p_t_1 = p.copy()  # p_{t-1} (posterior probabilities from previous iteration)
        p_t = np.zeros_like(p_t_1)

        for i in range(self.max_iter):
            p_t = p + 2 * W.T @ p_t_1  # Update rule
            if np.linalg.norm(p_t - p_t_1, ord=1) / np.linalg.norm(p_t, ord=1) < self.delta:
                if self.verbose:
                    print(f"Terminating after {i + 1} iterations")
                break
            p_t_1 = p_t.copy()

        p_t += 0.5  # shift back

        self.trust_values = -p_t  # p_t is posterior probability of being sybil

        return self.sybil_classification(p_t, self.threshold)

    def _initialize_priors(self):
        priors = np.full(self.graph.num_nodes(), 0.5)
        for i in range(self.graph.num_nodes()):
            if self.train_labels[i] == 1:
                priors[i] = 0.5 + self.theta
            if self.train_labels[i] == -1:
                priors[i] = 0.5 - self.theta

        return priors

    def _homophily_strength_matrix(self, adjacency_matrix) -> np.ndarray:
        num_nodes = self.graph.num_nodes()
        if self.variant == "C":  # SybilSCAR-C
            W = self.const_homophily_strength * adjacency_matrix
        elif self.variant == "D":  # SybilSCAR-D
            # W = np.zeros(shape=(num_nodes, num_nodes))
            # W = csr_matrix((num_nodes, num_nodes), dtype=float)
            row_list = []
            col_list = []
            data_list = []
            for (u, v) in self.graph.edges_list():
                d_u = self.graph.degree(u)
                W_vu = 1.0 / (2 * d_u)
                row_list.append(v)
                col_list.append(u)
                data_list.append(W_vu)

                d_v = self.graph.degree(v)
                W_uv = 1.0 / (2 * d_v)
                row_list.append(u)
                col_list.append(v)
                data_list.append(W_uv)

            row = np.array(row_list)
            col = np.array(col_list)
            data = np.array(data_list)
            W = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

            # W = W.tocsr()
        else:
            raise Exception("Variant not supported")  # This exception will be caught in constructor
        return W
