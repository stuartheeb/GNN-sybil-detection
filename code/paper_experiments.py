from benchmark import *
import seaborn as sns
from samplers import *
import os


RUN_EXPERIMENT = False
GENERATE_PLOT = True

class Experiment:
    def __init__(self, num_experiments: int, subdirectory: str, active: bool = True):
        self.num_experiments = num_experiments
        self.directory = f"paper_experiments/{subdirectory}"

        if not os.path.exists(f"{self.directory}/"):
            os.makedirs(f"{self.directory}/")

        self.readme = None
        self.active = active

        self.statistics = ["AUC", "accuracy", "F1"]
        self.seeds = list(range(42, 42 + num_experiments))
        self.data = None

    def run_experiment(self):
        print(f"Running experiment {self.__class__.__name__}")
        self._run_experiment()
        print(f"Experiment completed. Writing data to {self.directory}/data.csv")
        if self.data is not None:
            self.write_csv(self.data, "data.csv")
            print(f"Data written successfully.")
            text_file = open(f"{self.directory}/readme.txt", "w")
            text_file.write(f"{self.readme}\n\nSeeds used: {self.seeds}")
            text_file.close()

    def _run_experiment(self):
        raise NotImplementedError

    def plot_results(self):
        print(f"Plotting experiment {self.__class__.__name__}")
        if self.data is None:
            self.data = pd.read_csv(f"{self.directory}/data.csv", sep=" ")
        self._plot_results()
        print(f"Plots created successfully.")

    def _plot_results(self):
        raise NotImplementedError

    def write_csv(self, df: pd.DataFrame, file_name: str):
        if not os.path.exists(f"{self.directory}/"):
            os.makedirs(f"{self.directory}/")
        df.to_csv(f"{self.directory}/{file_name}", sep=" ", index=False)

    def _full_pretrain_algorithm_list(self):
        return [
            SybilFinderGAT(num_epochs=1000,
                           num_layers=2,
                           hidden_width=4,
                           num_heads=4,
                           patience=10,
                           name="SybilGAT_L2"),
            SybilFinderGAT(num_epochs=1000,
                           num_layers=4,
                           hidden_width=4,
                           num_heads=4,
                           patience=10,
                           name="SybilGAT_L4"),
            SybilFinderGAT(num_epochs=1000,
                           num_layers=8,
                           hidden_width=4,
                           num_heads=4,
                           patience=10,
                           name="SybilGAT_L8")
        ]


class PretrainOnSubgraphExperiment(Experiment):
    def __init__(self,
                 experiment_variants: [str],
                 num_experiments: int,
                 active: bool = True):
        super().__init__(num_experiments=num_experiments,
                         subdirectory="pretrain_on_subgraph",
                         active=active)
        self.readme = "Pretrain on a subgraph of the social network and evaluate on the remaining social network."

        self.experiment_variants = experiment_variants
        self.pretrain_algorithms = {variant: [] for variant in self.experiment_variants}
        for variant in self.experiment_variants:
            self.pretrain_algorithms[variant] = self._full_pretrain_algorithm_list()

        self.baseline_algorithms = [
            SybilRank(),
            SybilBelief(),
            SybilSCAR()
        ]

    def _run_experiment(self):
        data = []
        attack_edges_type, social_network_type = None, None

        total_num_experiments = len(self.experiment_variants) * self.num_experiments
        p_bar = tqdm(total=total_num_experiments)
        for variant in self.experiment_variants:
            experiment_data = np.zeros(shape=(len(self.pretrain_algorithms[variant]) + len(self.baseline_algorithms),
                                              len(self.statistics), self.num_experiments))

            # Set up social network (base network doesn't change for different experiments, only the sampling)
            if variant == "twitter":
                social_network = Twitter270K(is_directed=False)
                number_of_nodes = social_network.network.num_nodes() // 20
                attack_edges_per_sybil = None
                social_network_type = "Twitter"

            for i in range(self.num_experiments):
                random.seed(self.seeds[i])
                np.random.seed(self.seeds[i])
                torch.manual_seed(self.seeds[i])

                # Set up social network
                if variant == "twitter":
                    # Social network was already loaded
                    social_network_type = "Twitter"
                elif variant == "facebook-random" or variant == "facebook-targeted":
                    social_network = SocialNetworkFromRegions(honest_region=FacebookSNAP(),
                                                              sybil_region=FacebookSNAP(is_sybil=True))
                    number_of_nodes = social_network.network.num_nodes() // 10
                    attack_edges_per_sybil = 20
                    social_network_type = "Facebook"
                elif variant == "synthetic-L-random" or variant == "synthetic-L-targeted":
                    social_network = SocialNetworkFromRegions(honest_region=PowerLawGraph(25000, 5, 0.8),
                                                              sybil_region=PowerLawGraph(25000, 5, 0.8, is_sybil=True))
                    number_of_nodes = social_network.network.num_nodes() // 10
                    attack_edges_per_sybil = 8
                    social_network_type = "PowerLaw-Large"
                elif variant == "synthetic-S-random" or variant == "synthetic-S-targeted":
                    social_network = SocialNetworkFromRegions(honest_region=PowerLawGraph(5000, 5, 0.8),
                                                              sybil_region=PowerLawGraph(5000, 5, 0.8, is_sybil=True))
                    number_of_nodes = social_network.network.num_nodes() // 10
                    attack_edges_per_sybil = 8
                    social_network_type = "PowerLaw-Small"
                else:
                    raise Exception(f"Invalid experiment variant: {variant}")

                if "twitter" not in variant:
                    # Create train / test split (it is already done for Twitter dataset)
                    social_network.train_test_split(train_fraction=0.05)

                    # Set up attack
                    if "targeted" in variant:
                        attack = Attack(p_targeted=0.1, pdf_targeted=[0.25, 0.25, 0.5], seed=self.seeds[i])
                        attack_edges_type = "targeted"
                    else:
                        attack = RandomAttack(seed=self.seeds[i])
                        attack_edges_type = "random"
                    attack.perform_attack(social_network=social_network,
                                          num_attack_edges=attack_edges_per_sybil * social_network.sybil_region.num_nodes(),
                                          honest_targets=social_network.train_honest_nodes,
                                          sybil_targets=social_network.train_sybil_nodes)

                # Network sampling
                subsampled_social_network, remaining_social_network = ForestFireSampler(
                    seed=self.seeds[i]).sample_social_network(
                    social_network=social_network,
                    number_of_nodes=number_of_nodes,
                    num_sources=1,
                    verbose=False)

                for pretrain_algorithm in self.pretrain_algorithms[variant]:
                    # Put algorithm into training mode
                    pretrain_algorithm.train_model = True

                # Pretraining
                pretrain_evaluator = Evaluator(social_network=subsampled_social_network, verbose=False)
                pretrain_evaluator.evaluate_all(algorithms=self.pretrain_algorithms[variant])
                # pretrain_evaluator.get_all_stats()

                for pretrain_algorithm in self.pretrain_algorithms[variant]:
                    # Do not fine-tune, directly apply to remaining social network
                    pretrain_algorithm.train_model = False

                # Evaluation
                evaluator = Evaluator(social_network=remaining_social_network, verbose=False)
                evaluator.evaluate_all(algorithms=self.pretrain_algorithms[variant] + self.baseline_algorithms,
                                       reinitialize_GNNs=False)
                all_stats = evaluator.get_all_stats()
                for algo_id, algo in enumerate(self.pretrain_algorithms[variant] + self.baseline_algorithms):
                    for stat_id, stat in enumerate(self.statistics):
                        experiment_data[algo_id, stat_id, i] = all_stats[algo_id][stat]

                # Progress bar
                p_bar.update(n=1)
                p_bar.refresh()

            # Mean over experiments
            experiment_mean = np.mean(experiment_data, axis=2)
            experiment_std = np.std(experiment_data, axis=2)

            for algo_id, algo in enumerate(self.pretrain_algorithms[variant] + self.baseline_algorithms):
                data_point = {
                    "experiment": variant,
                    "algorithm": algo.name,
                    "attack_edges_type": attack_edges_type,
                    "social_network_type": social_network_type
                }
                for stat_id, stat in enumerate(self.statistics):
                    data_point[stat] = experiment_mean[algo_id, stat_id]
                    data_point[f"{stat}_std"] = experiment_std[algo_id, stat_id]
                    for i in range(self.num_experiments):
                        data_point[f"{stat}_{i}"] = experiment_data[algo_id, stat_id, i]
                data.append(data_point)
        self.data = pd.DataFrame(data)

    def _plot_results(self):
        for variant in self.experiment_variants:
            sns.set_theme()
            plt.figure(figsize=(10, 10))
            variant_data = self.data[self.data['experiment'] == variant]
            ax = sns.barplot(
                data=variant_data,
                x="algorithm",
                y="AUC",
                hue="algorithm",
                errorbar=None,
                width=0.5
            )

            for i, (mean, std) in enumerate(zip(variant_data['AUC'], variant_data['AUC_std'])):
                ax.text(i, mean, f"{round(mean, 4)} +- {round(std, 4)}", ha='center', va='bottom')

            plt.title(variant)
            plt.ylim(0, 1)
            plt.savefig(f"{self.directory}/barplot_{variant}{'_5exp' if self.num_experiments == 5 else ''}.pdf")


class PretrainOnSmallSyntheticSocialNetworkExperiment(Experiment):
    def __init__(self,
                 experiment_variants: [str],
                 num_experiments: int,
                 active: bool = True):
        super().__init__(num_experiments=num_experiments,
                         subdirectory="pretrain_on_small_synthetic_social_network",
                         active=active)
        self.readme = "Pretrain on a small synthetic social network and evaluate on the full social network."

        self.experiment_variants = experiment_variants
        self.pretrain_algorithms = {variant: [] for variant in self.experiment_variants}
        for variant in self.experiment_variants:
            self.pretrain_algorithms[variant] = self._full_pretrain_algorithm_list()

        self.baseline_algorithms = [
            SybilRank(),
            SybilBelief(),
            SybilSCAR()
        ]

    def _run_experiment(self):
        data = []
        attack_edges_type, small_social_network_type, large_social_network_type = None, None, None

        total_num_experiments = len(self.experiment_variants) * self.num_experiments
        p_bar = tqdm(total=total_num_experiments)
        for variant in self.experiment_variants:
            experiment_data = np.zeros(shape=(len(self.pretrain_algorithms[variant]) + len(self.baseline_algorithms),
                                              len(self.statistics), self.num_experiments))

            for i in range(self.num_experiments):
                random.seed(self.seeds[i])
                np.random.seed(self.seeds[i])
                torch.manual_seed(self.seeds[i])

                small_n = 1000
                large_n = 10000
                small_attack_edges_per_sybil = 8
                large_attack_edges_per_sybil = 8

                # Set up social network
                if variant == "erdos-renyi-random" or variant == "erdos-renyi-targeted":
                    small_social_network = SocialNetworkFromRegions(
                        honest_region=ErdosRenyiGraph(small_n, 2 * 6 / small_n),
                        sybil_region=ErdosRenyiGraph(small_n, 2 * 6 / small_n,
                                                     is_sybil=True))
                    large_social_network = SocialNetworkFromRegions(
                        honest_region=ErdosRenyiGraph(large_n, 2 * 6 / large_n),
                        sybil_region=ErdosRenyiGraph(large_n, 2 * 6 / large_n,
                                                     is_sybil=True))
                    small_social_network_type = "Erdos Renyi"
                    large_social_network_type = "Erdos Renyi"
                elif variant == "power-law-random" or variant == "power-law-targeted":
                    small_social_network = SocialNetworkFromRegions(
                        honest_region=PowerLawGraph(small_n, 6, 0.8),
                        sybil_region=PowerLawGraph(small_n, 6, 0.8,
                                                   is_sybil=True))
                    large_social_network = SocialNetworkFromRegions(
                        honest_region=PowerLawGraph(large_n, 6, 0.8),
                        sybil_region=PowerLawGraph(large_n, 6, 0.8,
                                                   is_sybil=True))
                    small_social_network_type = "Power Law"
                    large_social_network_type = "Power Law"
                elif variant == "barabasi-albert-random" or variant == "barabasi-albert-targeted":
                    small_social_network = SocialNetworkFromRegions(
                        honest_region=BarabasiAlbertGraph(small_n, 6),
                        sybil_region=BarabasiAlbertGraph(small_n, 6, is_sybil=True))
                    large_social_network = SocialNetworkFromRegions(
                        honest_region=BarabasiAlbertGraph(large_n, 6),
                        sybil_region=BarabasiAlbertGraph(large_n, 6, is_sybil=True))
                    small_social_network_type = "Barabasi Albert"
                    large_social_network_type = "Barabasi Albert"
                elif variant == "power-law-facebook-random" or variant == "power-law-facebook-targeted":
                    small_social_network = SocialNetworkFromRegions(
                        honest_region=PowerLawGraph(small_n, 6, 0.8),
                        sybil_region=PowerLawGraph(small_n, 6, 0.8, is_sybil=True))
                    large_social_network = SocialNetworkFromRegions(
                        honest_region=FacebookSNAP(),
                        sybil_region=FacebookSNAP(is_sybil=True)
                    )
                    small_social_network_type = "Power Law"
                    large_social_network_type = "Facebook"
                    small_attack_edges_per_sybil = 8
                    large_attack_edges_per_sybil = 20
                else:
                    raise Exception(f"Invalid experiment variant: {variant}")

                # Create train / test split
                small_social_network.train_test_split(train_fraction=0.05)
                large_social_network.train_test_split(train_fraction=0.05)

                # Set up attack
                if "targeted" in variant:
                    attack = Attack(p_targeted=0.1, pdf_targeted=[0.25, 0.25, 0.5], seed=self.seeds[i])
                    attack_edges_type = "targeted"
                else:
                    attack = RandomAttack(seed=self.seeds[i])
                    attack_edges_type = "random"
                attack.perform_attack(social_network=small_social_network,
                                      num_attack_edges=small_attack_edges_per_sybil * small_social_network.sybil_region.num_nodes(),
                                      honest_targets=small_social_network.train_honest_nodes,
                                      sybil_targets=small_social_network.train_sybil_nodes)
                attack.perform_attack(social_network=large_social_network,
                                      num_attack_edges=large_attack_edges_per_sybil * large_social_network.sybil_region.num_nodes(),
                                      honest_targets=large_social_network.train_honest_nodes,
                                      sybil_targets=large_social_network.train_sybil_nodes)

                for pretrain_algorithm in self.pretrain_algorithms[variant]:
                    # Put algorithm into training mode
                    pretrain_algorithm.train_model = True

                # Pretraining
                pretrain_evaluator = Evaluator(social_network=small_social_network, verbose=False)
                pretrain_evaluator.evaluate_all(algorithms=self.pretrain_algorithms[variant])
                pretrain_evaluator.get_all_stats()

                for pretrain_algorithm in self.pretrain_algorithms[variant]:
                    # Do not fine-tune, directly apply to remaining social network
                    pretrain_algorithm.train_model = False

                # Evaluation
                evaluator = Evaluator(social_network=large_social_network, verbose=False)
                evaluator.evaluate_all(algorithms=self.pretrain_algorithms[variant] + self.baseline_algorithms,
                                       reinitialize_GNNs=False)
                all_stats = evaluator.get_all_stats()
                for algo_id, algo in enumerate(self.pretrain_algorithms[variant] + self.baseline_algorithms):
                    for stat_id, stat in enumerate(self.statistics):
                        experiment_data[algo_id, stat_id, i] = all_stats[algo_id][stat]

                # Progress bar
                p_bar.update(n=1)
                p_bar.refresh()

            # Mean over experiments
            experiment_mean = np.mean(experiment_data, axis=2)
            experiment_std = np.std(experiment_data, axis=2)

            for algo_id, algo in enumerate(self.pretrain_algorithms[variant] + self.baseline_algorithms):
                data_point = {
                    "experiment": variant,
                    "algorithm": algo.name,
                    "attack_edges_type": attack_edges_type,
                    "pretrain_social_network_type": small_social_network_type,
                    "evaluate_social_network_type": large_social_network_type,
                    "small_n": 2 * small_n,
                    "large_n": 2 * large_n,
                    "small_attack_edges_per_sybil": small_attack_edges_per_sybil,
                    "large_attack_edges_per_sybil": large_attack_edges_per_sybil
                }
                for stat_id, stat in enumerate(self.statistics):
                    data_point[stat] = experiment_mean[algo_id, stat_id]
                    data_point[f"{stat}_std"] = experiment_std[algo_id, stat_id]
                    for i in range(self.num_experiments):
                        data_point[f"{stat}_{i}"] = experiment_data[algo_id, stat_id, i]
                data.append(data_point)
        self.data = pd.DataFrame(data)

    def _plot_results(self):
        for variant in self.experiment_variants:
            sns.set_theme()
            plt.figure(figsize=(10, 10))
            variant_data = self.data[self.data['experiment'] == variant]
            ax = sns.barplot(
                data=variant_data,
                x="algorithm",
                y="AUC",
                hue="algorithm",
                errorbar=None,
                width=0.5
            )

            for i, (mean, std) in enumerate(zip(variant_data['AUC'], variant_data['AUC_std'])):
                ax.text(i, mean, f"{round(mean, 4)} +- {round(std, 4)}", ha='center', va='bottom')

            plt.title(variant)
            plt.ylim(0, 1)
            plt.savefig(f"{self.directory}/barplot_{variant}{'_5exp' if self.num_experiments == 5 else ''}.pdf")


class AddingAttackEdgesAfterPretrainingExperiment(Experiment):
    def __init__(self,
                 experiment_variants: [str],
                 num_experiments: int,
                 active: bool = True):
        super().__init__(num_experiments=num_experiments,
                         subdirectory="adding_attack_edges_after_pretraining",
                         active=active)
        self.readme = "Pretrain on a subgraph of the social network, evaluate on the social network that now contains not only random but also targeted attack edges."

        self.experiment_variants = experiment_variants
        self.pretrain_algorithms = {variant: [] for variant in self.experiment_variants}
        for variant in self.experiment_variants:
            self.pretrain_algorithms[variant] = self._full_pretrain_algorithm_list()

        self.baseline_algorithms = [
            SybilRank(),
            SybilBelief(),
            SybilSCAR()
        ]

    def _run_experiment(self):
        data = []
        attack_edges_type, small_social_network_type, large_social_network_type = None, None, None

        total_num_experiments = len(self.experiment_variants) * self.num_experiments
        p_bar = tqdm(total=total_num_experiments)
        for variant in self.experiment_variants:
            experiment_data = np.zeros(shape=(len(self.pretrain_algorithms[variant]) + len(self.baseline_algorithms),
                                              len(self.statistics), self.num_experiments))

            for i in range(self.num_experiments):
                random.seed(self.seeds[i])
                np.random.seed(self.seeds[i])
                torch.manual_seed(self.seeds[i])

                n = 1000
                base_attack_edges_per_sybil = 8
                attacked_attack_edges_per_sybil = 8

                # Set up regions
                if variant == "erdos-renyi":
                    honest_region = ErdosRenyiGraph(n, 2 * 6 / n)
                    sybil_region = ErdosRenyiGraph(n, 2 * 6 / n, is_sybil=True)
                elif variant == "power-law":
                    honest_region = PowerLawGraph(n, 6, 0.8)
                    sybil_region = PowerLawGraph(n, 6, 0.8, is_sybil=True)
                elif variant == "barabasi-albert":
                    honest_region = BarabasiAlbertGraph(n, 6)
                    sybil_region = BarabasiAlbertGraph(n, 6, is_sybil=True)
                elif variant == "facebook":
                    honest_region = FacebookSNAP()
                    sybil_region = FacebookSNAP(is_sybil=True)
                    base_attack_edges_per_sybil = 20
                    attacked_attack_edges_per_sybil = 20
                else:
                    raise Exception(f"Invalid experiment variant: {variant}")

                # Set up social networks
                base_network = SocialNetworkFromRegions(honest_region=honest_region,
                                                        sybil_region=sybil_region)
                RandomAttack(seed=self.seeds[i]).perform_attack(social_network=base_network,
                                                                num_attack_edges=base_attack_edges_per_sybil * base_network.sybil_region.num_nodes())

                attacked_network = SocialNetworkFromRegions(honest_region=honest_region,
                                                            sybil_region=sybil_region)
                Attack(p_targeted=0.2, pdf_targeted=[0.5, 0.5], seed=self.seeds[i]).perform_attack(
                    social_network=attacked_network,
                    num_attack_edges=attacked_attack_edges_per_sybil * attacked_network.sybil_region.num_nodes())

                # Create train / test split
                base_network.train_test_split(train_fraction=0.05)
                attacked_network.train_test_split(train_fraction=0.05)

                for pretrain_algorithm in self.pretrain_algorithms[variant]:
                    # Put algorithm into training mode
                    pretrain_algorithm.train_model = True

                # Pretraining
                pretrain_evaluator = Evaluator(social_network=base_network, verbose=False)
                pretrain_evaluator.evaluate_all(algorithms=self.pretrain_algorithms[variant])
                pretrain_evaluator.get_all_stats()

                for pretrain_algorithm in self.pretrain_algorithms[variant]:
                    # Do not fine-tune, directly apply to remaining social network
                    pretrain_algorithm.train_model = False

                # Evaluation
                evaluator = Evaluator(social_network=attacked_network, verbose=False)
                evaluator.evaluate_all(algorithms=self.pretrain_algorithms[variant] + self.baseline_algorithms,
                                       reinitialize_GNNs=False)
                all_stats = evaluator.get_all_stats()
                for algo_id, algo in enumerate(self.pretrain_algorithms[variant] + self.baseline_algorithms):
                    for stat_id, stat in enumerate(self.statistics):
                        experiment_data[algo_id, stat_id, i] = all_stats[algo_id][stat]

                # Progress bar
                p_bar.update(n=1)
                p_bar.refresh()

            # Mean over experiments
            experiment_std = np.std(experiment_data, axis=2)
            experiment_mean = np.mean(experiment_data, axis=2)

            for algo_id, algo in enumerate(self.pretrain_algorithms[variant] + self.baseline_algorithms):
                data_point = {
                    "experiment": variant,
                    "algorithm": algo.name,
                    "attack_edges_type": attack_edges_type,
                    "n": 2 * n,
                    "base_attack_edges_per_sybil": base_attack_edges_per_sybil,
                    "attacked_attack_edges_per_sybil": attacked_attack_edges_per_sybil
                }
                for stat_id, stat in enumerate(self.statistics):
                    data_point[stat] = experiment_mean[algo_id, stat_id]
                    data_point[f"{stat}_std"] = experiment_std[algo_id, stat_id]
                    for i in range(self.num_experiments):
                        data_point[f"{stat}_{i}"] = experiment_data[algo_id, stat_id, i]
                data.append(data_point)
        self.data = pd.DataFrame(data)

    def _plot_results(self):
        for variant in self.experiment_variants:
            sns.set_theme()
            plt.figure(figsize=(10, 10))
            variant_data = self.data[self.data['experiment'] == variant]
            ax = sns.barplot(
                data=variant_data,
                x="algorithm",
                y="AUC",
                hue="algorithm",
                errorbar=None,
                width=0.5
            )

            for i, (mean, std) in enumerate(zip(variant_data['AUC'], variant_data['AUC_std'])):
                ax.text(i, mean, f"{round(mean, 4)} +- {round(std, 4)}", ha='center', va='bottom')

            plt.title(variant)
            plt.ylim(0, 1)
            plt.savefig(f"{self.directory}/barplot_{variant}{'_5exp' if self.num_experiments == 5 else ''}.pdf")


class GeneralPerformanceExperiment(Experiment):
    def __init__(self,
                 experiment_variants: [str],
                 num_experiments: int,
                 active: bool = True):
        super().__init__(num_experiments=num_experiments,
                         subdirectory="general_performance",
                         active=active)
        self.readme = "Comprehensive algorithm evaluation against multiple baselines, increasing the number of attack edges."

        self.experiment_variants = experiment_variants

        self.pretrain_algorithms = self._full_pretrain_algorithm_list()
        self.pretrain_algorithms = [self.pretrain_algorithms[0], self.pretrain_algorithms[1]]

        self.baseline_algorithms = [
            LegacyAlgorithm("SybilRank"),
            LegacyAlgorithm("SybilBelief"),
            SybilSCAR()
        ]

    def _run_experiment(self):
        num_attack_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # per sybil

        train_fractions = [
            (0.05, 0.05),
        ]

        n = 1000
        regions = []
        if "power-law" in self.experiment_variants:
            regions.append((
                PowerLawGraph(n, 6, 0.8),
                PowerLawGraph(n, 6, 0.8, is_sybil=True)
            ))
        if "barabasi-albert" in self.experiment_variants:
            regions.append((
                BarabasiAlbertGraph(n, 6),
                BarabasiAlbertGraph(n, 6, is_sybil=True)
            ))

        attacks = [
            RandomAttack(),
            # Attack(p_targeted=0.1, pdf_targeted=[0.5, 0.25, 0.25]),
            # Attack(p_targeted=0.1, pdf_targeted=[0.16, 0.34, 0.5]),
        ]

        for algorithm in self.pretrain_algorithms:
            algorithm.fine_tune = True

        benchmark = Benchmark(regions=regions,
                              algorithms=self.pretrain_algorithms + self.baseline_algorithms,
                              attacks=attacks,
                              num_attack_edges=num_attack_edges,
                              train_fractions=train_fractions,
                              num_experiments=self.num_experiments,
                              directory="experiment_robustness",
                              verbose=False,
                              seeds=self.seeds)
        self.data = benchmark.run_benchmark(statistics=self.statistics, write_csv=False)

    def _plot_results(self):
        sns.set_theme()
        for variant in self.experiment_variants:
            plt.figure(figsize=(5, 5))
            if variant == "power-law":
                variant_data = self.data[self.data["model"].str.contains("PowerLaw")]
                model_name = "PL"
            elif variant == "barabasi-albert":
                variant_data = self.data[self.data["model"].str.contains("BarabasiAlbert")]
                model_name = "BA"
            else:
                raise Exception(f"Invalid experiment variant: {variant}")
            sns.lineplot(
                data=variant_data,
                x="attack_edges_per_sybil",
                y="AUC",
                hue="algorithm"
            )
            # plt.title("Robustness evaluation")
            plt.ylim(0.4, 1)
            plt.ylabel("AUC")
            plt.xlabel("Attack edges per sybil")
            plt.savefig(f"{self.directory}/exp4_{model_name}.pdf")


NUM_EXPERIMENTS = 5

if __name__ == "__main__":
    experiments = [
        PretrainOnSubgraphExperiment(experiment_variants=["twitter",
                                                          "facebook-random",
                                                          "facebook-targeted",
                                                          "synthetic-S-random",
                                                          "synthetic-L-random",
                                                          ],
                                     num_experiments=NUM_EXPERIMENTS,
                                     active=True),  # DONE EVALUATING
        PretrainOnSmallSyntheticSocialNetworkExperiment(experiment_variants=["power-law-random",
                                                                             "power-law-targeted",
                                                                             "barabasi-albert-random",
                                                                             "barabasi-albert-targeted",
                                                                             "power-law-facebook-random",
                                                                             "power-law-facebook-targeted"
                                                                             ],
                                                        num_experiments=NUM_EXPERIMENTS,
                                                        active=True),  # DONE EVALUATAING
        AddingAttackEdgesAfterPretrainingExperiment(experiment_variants=["power-law",
                                                                         "barabasi-albert",
                                                                         "facebook"
                                                                         ],
                                                    num_experiments=NUM_EXPERIMENTS,
                                                    active=True),  # DONE EVALUATING
        GeneralPerformanceExperiment(experiment_variants=["power-law",
                                                          "barabasi-albert"
                                                          ],
                                     num_experiments=NUM_EXPERIMENTS, active=True)  # DONE EVALUATTING
    ]

    for experiment in experiments:
        if experiment.active:
            if RUN_EXPERIMENT:
                experiment.run_experiment()
            if GENERATE_PLOT:
                experiment.plot_results()
