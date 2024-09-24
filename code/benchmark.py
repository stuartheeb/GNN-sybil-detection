from evaluator import *
from social_networks import *
from attacks import *

from datetime import datetime

import pandas as pd

import os

from tqdm import tqdm

DEFAULT_STATS = ["AUC", "accuracy", "F1"]


class Benchmark:
    def __init__(self,
                 regions: [(Graph, Graph)],
                 algorithms: [SybilFinder],
                 attacks: [Attack],
                 num_attack_edges,
                 train_fractions: [(float, float)],
                 num_experiments: int = 1,
                 directory: str = "dir",
                 verbose: bool = True,
                 seeds: [int] = None):
        self.regions = regions
        self.algorithms = algorithms
        self.attacks = attacks
        self.num_attack_edges = num_attack_edges
        self.train_fractions = train_fractions
        self.num_experiments = num_experiments
        self.directory = directory
        self.verbose = verbose
        if seeds is None or len(seeds) != num_experiments:
            seeds = list(range(42, 42 + num_experiments))
        self.seeds = seeds

    def write_csv(self, df: pd.DataFrame, file_name: str):
        if not os.path.exists(f"{self.directory}/"):
            os.makedirs(f"{self.directory}/")
        df.to_csv(f"{self.directory}/{file_name}", sep=" ", index=False)

    def run_benchmark(self, statistics: [str] = None, write_csv: bool = True, file_name: str = None):

        if statistics is None:
            statistics = DEFAULT_STATS

        data = []

        total_num_experiments = len(self.regions) * len(self.attacks) * len(self.num_attack_edges) * len(
            self.train_fractions) * self.num_experiments
        if self.verbose:
            print(f"\n Total number of experiments: {total_num_experiments}\n")

        # Progress bar
        p_bar = tqdm(total=total_num_experiments)

        for honest_region, sybil_region in self.regions:
            for attack in self.attacks:
                for num_attack_edges in self.num_attack_edges:
                    if isinstance(num_attack_edges, float) or isinstance(num_attack_edges, int):
                        # Attack edges per sybil
                        num_attack_edges = [int(num_attack_edges * sybil_region.num_nodes()), 0, 0]

                    for train_fraction_honest, train_fraction_sybil in self.train_fractions:

                        experiments_data = np.zeros(
                            shape=(len(self.algorithms), len(statistics), self.num_experiments))
                        n_H_train = None
                        n_H_test = None
                        n_S_train = None
                        n_S_test = None
                        for experiment_id in range(self.num_experiments):

                            random.seed(self.seeds[experiment_id])
                            np.random.seed(self.seeds[experiment_id])
                            torch.manual_seed(self.seeds[experiment_id])

                            social_network = SocialNetworkFromRegions(honest_region=honest_region,
                                                                      sybil_region=sybil_region)

                            train_honest, train_sybil, test_honest, test_sybil = social_network.train_test_split(
                                train_fraction_honest=train_fraction_honest,
                                train_fraction_sybil=train_fraction_sybil)
                            n_H_train = len(train_honest)
                            n_S_train = len(train_sybil)
                            n_H_test = len(test_honest)
                            n_S_test = len(test_sybil)

                            # Targets are overridden by the attack class if applicable
                            attack.perform_attack(social_network=social_network,
                                                  honest_targets=train_honest,
                                                  sybil_targets=social_network.sybil_nodes,
                                                  # TODO change back to train_sybil ??? then: problem with max number of attack edges
                                                  # sybil_targets=train_sybil,
                                                  num_attack_edges=num_attack_edges)

                            evaluator = Evaluator(social_network=social_network,
                                                  train_honest_nodes=train_honest,
                                                  train_sybil_nodes=train_sybil,
                                                  verbose=self.verbose)

                            evaluator.evaluate_all(algorithms=self.algorithms)
                            all_stats = evaluator.get_all_stats()

                            algo_id = 0
                            for _ in self.algorithms:
                                stat_id = 0
                                for stat in statistics:
                                    experiments_data[algo_id, stat_id, experiment_id] = all_stats[algo_id][stat]
                                    stat_id += 1
                                algo_id += 1

                            # Progress bar
                            p_bar.update(n=1)
                            p_bar.refresh()

                        # Average over experiments
                        experiments_data = np.mean(experiments_data, axis=2)

                        algo_id = 0
                        for algorithm in self.algorithms:
                            data_point = {
                                "honest_model": honest_region.name,
                                "sybil_model": sybil_region.name,
                                "model": honest_region.name if honest_region.name == sybil_region.name else f"{honest_region.name} - {sybil_region.name}",
                                "n_H": honest_region.num_nodes(),
                                "m_H": honest_region.num_edges(),
                                "n_S": sybil_region.num_nodes(),
                                "m_S": sybil_region.num_edges(),
                                "n": algorithm.graph.num_nodes(),
                                "m": algorithm.graph.num_edges(),
                                "type": "directed" if algorithm.graph.is_directed() else "undirected",
                                "algorithm": algorithm.name,
                                "attack": attack.name,
                                "attack_edges_bi": num_attack_edges[0],
                                "attack_edges_to_sybil": num_attack_edges[1],
                                "attack_edges_to_honest": num_attack_edges[2],
                                "attack_edges_total": sum(num_attack_edges),
                                "attack_edges_per_sybil": sum(num_attack_edges) / sybil_region.num_nodes(),
                                "train_fraction_honest": train_fraction_honest,
                                "train_fraction_sybil": train_fraction_sybil,
                                "n_H_train": n_H_train,
                                "n_H_test": n_H_test,
                                "n_S_train": n_S_train,
                                "n_S_test": n_S_test
                            }
                            stat_id = 0
                            for stat in statistics:
                                data_point[stat] = experiments_data[algo_id, stat_id]
                                stat_id += 1

                            data.append(data_point)

                            algo_id += 1
        df = pd.DataFrame(data)

        if write_csv:
            if file_name is None:
                file_name = f"experiment {datetime.now().strftime('%Y-%m-%d %H %M %S')}.csv"
            self.write_csv(df, file_name)

        return df


class PretrainBenchmark(Benchmark):
    def __init__(self,
                 pretrain_social_network: SocialNetwork,
                 evaluation_social_network: SocialNetwork,
                 pretrain_algorithms: [SybilFinder],
                 baseline_algorithms: [SybilFinder],
                 num_experiments: int = 1,
                 directory: str = "dir",
                 verbose: bool = True,
                 seeds: [int] = None):
        super().__init__(regions=None,
                         algorithms=None,
                         attacks=None,
                         num_attack_edges=None,
                         train_fractions=None,
                         num_experiments=num_experiments,
                         directory=directory,
                         verbose=verbose,
                         seeds=seeds)
        self.pretrain_social_network = pretrain_social_network
        self.evaluation_social_network = evaluation_social_network
        self.pretrain_algorithms = pretrain_algorithms
        self.baseline_algorithms = baseline_algorithms

    def run_benchmark(self, statistics: [str] = None, write_csv: bool = True, file_name: str = None):
        if statistics is None:
            statistics = DEFAULT_STATS

        data = []

        total_num_experiments = len(self.regions) * len(self.attacks) * len(self.num_attack_edges) * len(
            self.train_fractions) * self.num_experiments
        if self.verbose:
            print(f"\n Total number of experiments: {total_num_experiments}\n")

        # Pretraining
