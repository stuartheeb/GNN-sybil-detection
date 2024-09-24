from evaluator import *
from samplers import *
from thesis_experiments_utils import *
from latex_utils import *
from tqdm import tqdm

NUM_EXPERIMENTS = 5
SEEDS = np.linspace(42, 42 + NUM_EXPERIMENTS - 1, NUM_EXPERIMENTS, dtype=int)
STATISTICS = ["AUC", "accuracy", "precision", "recall", "F1", "runtime", "pretrain_runtime"]

RUN = False
PLOT = True
TABLES = True

FINE_TUNE_EPOCHS = 0

EXPERIMENT_1_1 = True  # Pretraining Strategies: Subsampled Network
EXPERIMENT_1_2 = True  # Pretraining Strategies: Smaller Network
EXPERIMENT_2 = True  # Robustness
EXPERIMENT_3 = True  # Twitter
EXPERIMENT_4_1 = True  # Adv: Robustness, Targeted Attack Edges
EXPERIMENT_4_2 = True  # Adv: Pretraining before Targeted Attack
EXPERIMENT_5_1 = True  # Misc: Scaling Network Size
EXPERIMENT_5_2 = True  # Misc: Different Region Sizes
EXPERIMENT_5_3 = True  # Misc: Other Metrics
EXPERIMENT_5_4 = True  # Misc: Runtime

color_map = {
    'SybilRank': '#FEB24C',
    'SybilBelief': '#FC4E2A',
    'SybilSCAR': '#E31A1C',
    'SybilGCN-L2': '#9EC9E2',
    'SybilGCN-L4': '#3C93C2',
    'SybilGCN-L8': '#0D4A70',
    'SybilRGCN-L2': '#AF58BA',
    'SybilGAT-L2': '#9CCEA7',
    'SybilGAT-L4': '#40AD5A',
    'SybilGAT-L8': '#06592A',
    '\\textsc{SybilRank}': '#FEB24C',
    '\\textsc{SybilBelief}': '#FC4E2A',
    '\\textsc{SybilSCAR}': '#E31A1C',
    '\\textsc{SybilGCN-L2}': '#9EC9E2',
    '\\textsc{SybilGCN-L4}': '#3C93C2',
    '\\textsc{SybilGCN-L8}': '#0D4A70',
    '\\textsc{SybilRGCN-L2}': '#AF58BA',
    '\\textsc{SybilGAT-L2}': '#9CCEA7',
    '\\textsc{SybilGAT-L4}': '#40AD5A',
    '\\textsc{SybilGAT-L8}': '#06592A'
}

baseline_algorithms = [
    SybilRank(),
    SybilBelief(),
    SybilSCAR()
]

gnn_algorithms = [
    # SybilFinderGCN(num_epochs=1000,
    #                num_layers=2,
    #                patience=5,
    #                dropout=False,
    #                name="SybilGCN-L2"),
    # SybilFinderGCN(num_epochs=1000,
    #                num_layers=4,
    #                patience=5,
    #                dropout=False,
    #                name="SybilGCN-L4"),
    # SybilFinderGCN(num_epochs=1000,
    #                num_layers=8,
    #                patience=5,
    #                dropout=False,
    #                name="SybilGCN-L8"),
    # SybilFinderRGCN(num_epochs=1000,
    #                 num_layers=2,
    #                 patience=5,
    #                 dropout=False,
    #                 case_mask={"H-H": True, "S-S": True, "H-U": False, "S-U": False, "H-S": False},
    #                 name="SybilRGCN-L2"),
    SybilFinderGAT(num_epochs=1000,
                   num_layers=2,
                   patience=5,
                   dropout=True,
                   name="SybilGAT-L2"),
    # SybilFinderGAT(num_epochs=1000,
    #                num_layers=4,
    #                patience=5,
    #                dropout=True,
    #                name="SybilGAT-L4"),
    # SybilFinderGAT(num_epochs=1000,
    #                num_layers=8,
    #                patience=5,
    #                dropout=True,
    #                name="SybilGAT-L8")
]

NUM_ALGORITHMS = len(baseline_algorithms) + len(gnn_algorithms)

figures_directory = "thesis_experiments/thesis_figures"

print("Running thesis experiments...")

if EXPERIMENT_1_1:
    directory = "thesis_experiments/experiment_1_1"
    variants = ["facebook", "barabasi-albert", "power-law"]
    if RUN:
        print("Running thesis experiment 1.1")
        data = []
        raw_data = []

        total_counter = len(variants) * NUM_EXPERIMENTS
        p_bar = tqdm(total=total_counter)

        experiment_data = np.zeros(shape=(len(variants), NUM_ALGORITHMS,
                                          len(STATISTICS), NUM_EXPERIMENTS))
        algorithms = None
        for var_id, variant in enumerate(variants):
            for i in range(NUM_EXPERIMENTS):
                seed = int(SEEDS[i])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                algorithms = combine_and_copy_lists(baseline_algorithms, gnn_algorithms)

                if variant == "facebook":
                    social_network = SocialNetworkFromRegions(honest_region=FacebookSNAP(),
                                                              sybil_region=FacebookSNAP(is_sybil=True))
                    attack_edges_per_sybil = 20
                elif variant == "power-law":
                    social_network = SocialNetworkFromRegions(honest_region=PowerLawGraph(2500, 10, 0.8),
                                                              sybil_region=PowerLawGraph(2500, 10, 0.8, is_sybil=True))
                    attack_edges_per_sybil = 15
                elif variant == "barabasi-albert":
                    social_network = SocialNetworkFromRegions(honest_region=BarabasiAlbertGraph(2500, 10),
                                                              sybil_region=BarabasiAlbertGraph(2500, 10, is_sybil=True))
                    attack_edges_per_sybil = 15
                else:
                    raise Exception("Invalid variant")
                social_network.train_test_split(train_fraction=0.05)
                number_of_nodes = social_network.network.num_nodes() // 10

                RandomAttack().perform_attack(social_network=social_network,
                                              num_attack_edges=attack_edges_per_sybil * social_network.sybil_region.num_nodes(),
                                              honest_targets=social_network.train_honest_nodes,
                                              sybil_targets=social_network.train_sybil_nodes)

                # Network sampling
                subsampled_social_network, remaining_social_network = ForestFireSampler(
                    seed=seed).sample_social_network(
                    social_network=social_network,
                    number_of_nodes=number_of_nodes,
                    num_sources=1,
                    verbose=False)

                pretrain_algorithms = []
                for algorithm in algorithms:
                    if isinstance(algorithm, SybilGNN):
                        # Put algorithm into training mode
                        algorithm.train_model = True
                        algorithm.fine_tune = False
                        pretrain_algorithms.append(algorithm)

                pretrain_evaluator = Evaluator(social_network=subsampled_social_network,
                                               verbose=False)
                pretrain_evaluator.evaluate_all(algorithms=pretrain_algorithms)

                # for algorithm in pretrain_algorithms:
                #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
                #                     f"{directory}/training_curves/{variant}_{algorithm}_exp{i}_pretrain.pdf")

                # pretrain_evaluator.get_all_stats()

                for algorithm in algorithms:
                    if isinstance(algorithm, SybilGNN):
                        algorithm.pretrain_runtime = algorithm.runtime
                        # Do not fine-tune, directly apply to remaining social network
                        # algorithm.train_model = False
                        # Fine-tune
                        algorithm.fine_tune = True
                        algorithm.num_epochs = FINE_TUNE_EPOCHS
                        algorithm.patience = 3

                # Evaluation
                evaluator = Evaluator(social_network=remaining_social_network,
                                      verbose=False)
                evaluator.evaluate_all(algorithms=algorithms,
                                       reinitialize_GNNs=False)

                # for algorithm in pretrain_algorithms:
                #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
                #                     f"{directory}/training_curves/{variant}_{algorithm}_exp{i}_eval.pdf")

                all_stats = evaluator.get_all_stats()
                for algo_id, algo in enumerate(algorithms):
                    data_point = {
                        "variant": variant,
                        "algorithm": algo.name,
                        "seed": seed
                    }
                    for stat_id, stat in enumerate(STATISTICS):
                        experiment_data[var_id, algo_id, stat_id, i] = all_stats[algo_id][stat]
                        data_point[stat] = all_stats[algo_id][stat]
                    raw_data.append(data_point)
                    df_raw = pd.DataFrame(raw_data)
                    write_experiment_csv(df_raw, directory, "raw_data.csv")

                # Progress bar
                p_bar.update(n=1)
                p_bar.refresh()

        experiment_mean = np.mean(experiment_data, axis=3)
        experiment_std = np.std(experiment_data, axis=3)

        for var_id, variant in enumerate(variants):
            for algo_id, algo in enumerate(algorithms):
                data_point = {
                    "variant": variant,
                    "algorithm": algo.name
                }
                for stat_id, stat in enumerate(STATISTICS):
                    data_point[f"{stat}_mean"] = experiment_mean[var_id, algo_id, stat_id]
                    data_point[f"{stat}_std"] = experiment_std[var_id, algo_id, stat_id]
                data.append(data_point)

        df = pd.DataFrame(data)
        write_experiment_csv(df, directory)

        df_raw = pd.DataFrame(raw_data)
        write_experiment_csv(df_raw, directory, "raw_data.csv")
    if PLOT:
        print("Plotting thesis experiment 1.1")
        sns.set_theme()
        df = read_experiment_csv(directory)
        df_raw = read_experiment_csv(directory, "raw_data.csv")

        plt.figure(figsize=(10, 4))
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        ax = sns.barplot(
            x="variant",
            y="AUC",
            hue="algorithm",
            data=df_raw,
            capsize=0.1,
            err_kws={'linewidth': 1, 'color': 'black'},
            errorbar="sd",
            palette=color_map
        )

        legend = plt.legend(title="\\textbf{Algorithm}", bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_edgecolor('none')

        plt.ylim(0.4, 1)

        plt.ylabel("AUC")
        plt.xlabel(None)

        plt.tight_layout()
        plt.savefig(f"{directory}/exp1_1.pdf")
        plt.savefig(f"{figures_directory}/exp1_1.pdf")
        plt.close()

        write_latex_figure_file(figures_directory,
                                "exp1_1.pdf",
                                width=1.0,
                                caption="Evaluation of AUC score on the remaining network after pre-training on a sampled subgraph, consisting of 10\% of the nodes. Networks are constructed using the Facebook (FB) graph, Barabási-Albert (BA) model and Power Law (PL) model as regions, adding random attack edges.",
                                label="exp1_1")
    if TABLES:
        print("Making tables for thesis experiment 1.1")
        df = read_experiment_csv(directory)
        df = reformat_dataframe(df)

        latex_table = df.to_latex(
            columns=["variant", "algorithm", "AUC", "accuracy", "precision", "recall", "pretrain_runtime",
                     "runtime"],
            header=["Variant", "Algorithm", "AUC", "Accuracy", "Precision", "Recall", "Pretrain Runtime",
                    "Runtime"],
            index=False,
            float_format="{:.3f}".format
        )
        latex_table = latex_table.replace("nan ± nan", "")
        latex_table = latex_table.replace("\\textbackslash textsc", "\\textsc")
        latex_table = latex_table.replace("\\{", "{")
        latex_table = latex_table.replace("\\}", "}")

        latex_table_file = r"""
        \begin{table}
        \centering
        \caption{Complete data for the experiments from \Cref{sec:pretraining-sampled-subgraph}. Runtimes are in seconds. Five runs for each experiment.}
        \label{tab:exp1_1_full_data}
        \small % or \footnotesize for even smaller text
        \begin{adjustbox}{width=\textwidth}
        """ + latex_table + r"""
        \end{adjustbox}
        \end{table}
        """

        with open('thesis_experiments/thesis_tables/exp1_1_full_data.tex', 'w') as f:
            f.write(latex_table_file)

if EXPERIMENT_1_2:
    variants = ["BA-BA", "BA-PL", "BA-FB", "PL-BA", "PL-PL", "PL-FB"]
    directory = "thesis_experiments/experiment_1_2"
    if RUN:
        print("Running thesis experiment 1.2")
        data = []
        raw_data = []

        total_counter = len(variants) * NUM_EXPERIMENTS
        p_bar = tqdm(total=total_counter)

        experiment_data = np.zeros(shape=(len(variants), NUM_ALGORITHMS,
                                          len(STATISTICS), NUM_EXPERIMENTS))
        algorithms = None
        for var_id, variant in enumerate(variants):
            for i in range(NUM_EXPERIMENTS):
                seed = int(SEEDS[i])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                algorithms = combine_and_copy_lists(baseline_algorithms, gnn_algorithms)

                pretrain_size = 1000
                eval_size = 10000

                pretrain_attack_edges_per_sybil = 8
                eval_attack_edges_per_sybil = 8
                if variant == "BA-BA":
                    pretrain_social_network = SocialNetworkFromRegions(
                        honest_region=BarabasiAlbertGraph(pretrain_size, 6),
                        sybil_region=BarabasiAlbertGraph(pretrain_size, 6,
                                                         is_sybil=True))
                    eval_social_network = SocialNetworkFromRegions(
                        honest_region=BarabasiAlbertGraph(eval_size, 6),
                        sybil_region=BarabasiAlbertGraph(eval_size, 6,
                                                         is_sybil=True))
                elif variant == "BA-PL":
                    pretrain_social_network = SocialNetworkFromRegions(
                        honest_region=BarabasiAlbertGraph(pretrain_size, 6),
                        sybil_region=BarabasiAlbertGraph(pretrain_size, 6,
                                                         is_sybil=True))
                    eval_social_network = SocialNetworkFromRegions(
                        honest_region=PowerLawGraph(eval_size, 6, 0.8),
                        sybil_region=PowerLawGraph(eval_size, 6, 0.8,
                                                   is_sybil=True))
                elif variant == "BA-FB":
                    pretrain_social_network = SocialNetworkFromRegions(
                        honest_region=BarabasiAlbertGraph(pretrain_size, 6),
                        sybil_region=BarabasiAlbertGraph(pretrain_size, 6,
                                                         is_sybil=True))
                    eval_social_network = SocialNetworkFromRegions(
                        honest_region=FacebookSNAP(),
                        sybil_region=FacebookSNAP(is_sybil=True))
                    eval_attack_edges_per_sybil = 20
                elif variant == "PL-BA":
                    pretrain_social_network = SocialNetworkFromRegions(
                        honest_region=PowerLawGraph(pretrain_size, 6, 0.8),
                        sybil_region=PowerLawGraph(pretrain_size, 6, 0.8,
                                                   is_sybil=True))
                    eval_social_network = SocialNetworkFromRegions(
                        honest_region=BarabasiAlbertGraph(eval_size, 6),
                        sybil_region=BarabasiAlbertGraph(eval_size, 6,
                                                         is_sybil=True))
                elif variant == "PL-PL":
                    pretrain_social_network = SocialNetworkFromRegions(
                        honest_region=PowerLawGraph(pretrain_size, 6, 0.8),
                        sybil_region=PowerLawGraph(pretrain_size, 6, 0.8,
                                                   is_sybil=True))
                    eval_social_network = SocialNetworkFromRegions(
                        honest_region=PowerLawGraph(eval_size, 6, 0.8),
                        sybil_region=PowerLawGraph(eval_size, 6, 0.8,
                                                   is_sybil=True))
                elif variant == "PL-FB":
                    pretrain_social_network = SocialNetworkFromRegions(
                        honest_region=PowerLawGraph(pretrain_size, 6, 0.8),
                        sybil_region=PowerLawGraph(pretrain_size, 6, 0.8,
                                                   is_sybil=True))
                    eval_social_network = SocialNetworkFromRegions(
                        honest_region=FacebookSNAP(),
                        sybil_region=FacebookSNAP(is_sybil=True))
                    eval_attack_edges_per_sybil = 20
                else:
                    raise Exception("Invalid variant")

                pretrain_social_network.train_test_split(train_fraction=0.05)
                eval_social_network.train_test_split(train_fraction=0.05)

                RandomAttack().perform_attack(social_network=pretrain_social_network,
                                              num_attack_edges=pretrain_attack_edges_per_sybil * pretrain_social_network.sybil_region.num_nodes(),
                                              honest_targets=pretrain_social_network.train_honest_nodes,
                                              sybil_targets=pretrain_social_network.train_sybil_nodes)

                RandomAttack().perform_attack(social_network=eval_social_network,
                                              num_attack_edges=eval_attack_edges_per_sybil * eval_social_network.sybil_region.num_nodes(),
                                              honest_targets=eval_social_network.train_honest_nodes,
                                              sybil_targets=eval_social_network.train_sybil_nodes)

                pretrain_algorithms = []
                for algorithm in algorithms:
                    if isinstance(algorithm, SybilGNN):
                        # Put algorithm into training mode
                        algorithm.train_model = True
                        algorithm.fine_tune = False
                        pretrain_algorithms.append(algorithm)

                pretrain_evaluator = Evaluator(social_network=pretrain_social_network,
                                               verbose=False)
                pretrain_evaluator.evaluate_all(algorithms=pretrain_algorithms)

                # for algorithm in pretrain_algorithms:
                #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
                #    f"{directory}/training_curves/{variant}_{algorithm}_exp{i}_pretrain.pdf")

                # pretrain_evaluator.get_all_stats()

                for algorithm in algorithms:
                    if isinstance(algorithm, SybilGNN):
                        algorithm.pretrain_runtime = algorithm.runtime
                        # Do not fine-tune, directly apply to remaining social network
                        # algorithm.train_model = False
                        # Fine-tune
                        algorithm.fine_tune = True
                        algorithm.num_epochs = FINE_TUNE_EPOCHS
                        algorithm.patience = 3

                # Evaluation
                evaluator = Evaluator(social_network=eval_social_network,
                                      verbose=False)
                evaluator.evaluate_all(algorithms=algorithms,
                                       reinitialize_GNNs=False)

                # for algorithm in pretrain_algorithms:
                #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
                #                     f"{directory}/training_curves/{variant}_{algorithm}_exp{i}_eval.pdf")

                all_stats = evaluator.get_all_stats()
                for algo_id, algo in enumerate(algorithms):
                    data_point = {
                        "variant": variant,
                        "algorithm": algo.name,
                        "seed": seed
                    }
                    for stat_id, stat in enumerate(STATISTICS):
                        experiment_data[var_id, algo_id, stat_id, i] = all_stats[algo_id][stat]
                        data_point[stat] = all_stats[algo_id][stat]
                    raw_data.append(data_point)
                    df_raw = pd.DataFrame(raw_data)
                    write_experiment_csv(df_raw, directory, "raw_data.csv")

                # Progress bar
                p_bar.update(n=1)
                p_bar.refresh()

        experiment_mean = np.mean(experiment_data, axis=3)
        experiment_std = np.std(experiment_data, axis=3)

        for var_id, variant in enumerate(variants):
            for algo_id, algo in enumerate(algorithms):
                data_point = {
                    "variant": variant,
                    "algorithm": algo.name
                }
                for stat_id, stat in enumerate(STATISTICS):
                    data_point[f"{stat}_mean"] = experiment_mean[var_id, algo_id, stat_id]
                    data_point[f"{stat}_std"] = experiment_std[var_id, algo_id, stat_id]
                data.append(data_point)

        df = pd.DataFrame(data)
        write_experiment_csv(df, directory)

        df_raw = pd.DataFrame(raw_data)
        write_experiment_csv(df_raw, directory, "raw_data.csv")
    if PLOT:
        print("Plotting thesis experiment 1.2")
        sns.set_theme()
        df_raw = read_experiment_csv(directory, "raw_data.csv")

        for variant in ["BA", "PL"]:
            df_raw_filtered = df_raw[df_raw['variant'].isin([f"{variant}-BA", f"{variant}-PL", f"{variant}-FB"])]

            plt.figure(figsize=(10, 4))
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Computer Modern Roman']
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

            ax = sns.barplot(
                x="variant",
                y="AUC",
                hue="algorithm",
                data=df_raw_filtered,
                capsize=0.1,
                err_kws={'linewidth': 1, 'color': 'black'},
                errorbar="sd",
                palette=color_map
            )

            legend = plt.legend(title="\\textbf{Algorithm}", bbox_to_anchor=(1.05, 1), loc='upper left')
            legend.get_frame().set_facecolor('none')
            legend.get_frame().set_edgecolor('none')

            # plt.ylim(0.4, 1)

            plt.ylabel("AUC")
            plt.xlabel(None)

            plt.tight_layout()
            plt.savefig(f"{directory}/exp1_2_{variant}.pdf")
            plt.savefig(f"{figures_directory}/exp1_2_{variant}.pdf")
            plt.close()

            if variant == "BA":
                caption = "Evaluation of AUC score on a large network (FB, BA, PL) after pre-training on a small network constructed using the Barabási-Albert (BA) model, adding 8 random attack edges per Sybil in each case."
            elif variant == "PL":
                caption = "Evaluation of AUC score on a large network (FB, BA, PL) after pre-training on a small network constructed using the Power Law (PL) model, adding 8 random attack edges per Sybil in each case."
            else:
                raise Exception("Invalid variant")

            write_latex_figure_file(figures_directory,
                                    f"exp1_2_{variant}.pdf",
                                    width=1.0,
                                    caption=caption,
                                    label=f"exp1_2_{variant}")
    if TABLES:
        print("Making tables for thesis experiment 1.2")
        df = read_experiment_csv(directory)
        df = reformat_dataframe(df)

        latex_table = df.to_latex(
            columns=["variant", "algorithm", "AUC", "accuracy", "precision", "recall", "pretrain_runtime",
                     "runtime"],
            header=["Variant", "Algorithm", "AUC", "Accuracy", "Precision", "Recall", "Pretrain Runtime",
                    "Runtime"],
            index=False,
            float_format="{:.3f}".format
        )
        latex_table = latex_table.replace("nan ± nan", "")
        latex_table = latex_table.replace("\\textbackslash textsc", "\\textsc")
        latex_table = latex_table.replace("\\{", "{")
        latex_table = latex_table.replace("\\}", "}")

        latex_table_file = r"""
                \begin{table}
                \centering
                \caption{Complete data for the experiments from \Cref{sec:pretraining-smaller-network}. Runtimes are in seconds. Five runs for each experiment.}
                \label{tab:exp1_2_full_data}
                \small % or \footnotesize for even smaller text
                \begin{adjustbox}{width=\textwidth}
                """ + latex_table + r"""
                \end{adjustbox}
                \end{table}
                """

        with open('thesis_experiments/thesis_tables/exp1_2_full_data.tex', 'w') as f:
            f.write(latex_table_file)

if EXPERIMENT_2:
    variants = ["data_model_column", "graph_size_column", "train_nodes_fraction_column", "label_noise_fraction_column"]
    directory = "thesis_experiments/experiment_2"
    if RUN:
        print("Running thesis experiment 2")

        attack_edges = [2, 4, 6, 8, 10, 12, 14, 16]

        models_default = ["PL"]  # Default
        models = models_default
        graph_sizes_default = [1000]  # Default
        graph_sizes = graph_sizes_default
        train_set_sizes_default = [0.05]  # Default
        train_set_sizes = train_set_sizes_default
        label_noises_default = [0.0]  # Default
        label_noises = label_noises_default

        algorithms = None
        for var_id, variant in enumerate(variants):
            data = []
            raw_data = []
            if variant == "data_model_column":
                models = ["BA", "PL"]
                graph_sizes = graph_sizes_default
                train_set_sizes = train_set_sizes_default
                label_noises = label_noises_default
            elif variant == "graph_size_column":
                models = models_default
                graph_sizes = [500, 1000, 2000, 4000, 8000]
                train_set_sizes = train_set_sizes_default
                label_noises = label_noises_default
            elif variant == "train_nodes_fraction_column":
                models = models_default
                graph_sizes = graph_sizes_default
                train_set_sizes = [0.005, 0.01, 0.05, 0.1, 0.2]
                label_noises = label_noises_default
            elif variant == "label_noise_fraction_column":
                models = models_default
                graph_sizes = graph_sizes_default
                train_set_sizes = train_set_sizes_default
                label_noises = [0.0, 0.1, 0.2, 0.3]
            else:
                raise Exception("Unknown variant.")

            experiment_data = np.zeros(shape=(len(models),
                                              len(graph_sizes),
                                              len(train_set_sizes),
                                              len(label_noises),
                                              len(attack_edges),
                                              NUM_ALGORITHMS,
                                              len(STATISTICS),
                                              NUM_EXPERIMENTS)
                                       )

            print(f"Running variant: {variant}")

            total_counter = len(models) * len(graph_sizes) * len(train_set_sizes) * len(label_noises) * len(
                attack_edges) * NUM_EXPERIMENTS
            p_bar = tqdm(total=total_counter)

            for model_id, model in enumerate(models):
                for graph_size_id, graph_size in enumerate(graph_sizes):
                    for train_set_size_id, train_set_size in enumerate(train_set_sizes):
                        for label_noise_id, label_noise in enumerate(label_noises):
                            for att_id, attack_edges_per_sybil in enumerate(attack_edges):
                                for i in range(NUM_EXPERIMENTS):
                                    seed = int(SEEDS[i])
                                    random.seed(seed)
                                    np.random.seed(seed)
                                    torch.manual_seed(seed)

                                    algorithms = combine_and_copy_lists(baseline_algorithms, gnn_algorithms)

                                    if model == "facebook":
                                        honest_region = FacebookSNAP()
                                        sybil_region = FacebookSNAP(is_sybil=True)
                                    elif model == "BA":
                                        honest_region = BarabasiAlbertGraph(graph_size, 6)
                                        sybil_region = BarabasiAlbertGraph(graph_size, 6,
                                                                           is_sybil=True)
                                    elif model == "PL":
                                        honest_region = PowerLawGraph(graph_size, 6, 0.8)
                                        sybil_region = PowerLawGraph(graph_size, 6, 0.8, is_sybil=True)
                                    else:
                                        raise Exception("Unknown model.")

                                    social_network = SocialNetworkFromRegions(honest_region=honest_region,
                                                                              sybil_region=sybil_region)
                                    social_network.train_test_split(train_fraction=train_set_size)
                                    RandomAttack().perform_attack(social_network=social_network,
                                                                  num_attack_edges=attack_edges_per_sybil * social_network.sybil_region.num_nodes(),
                                                                  honest_targets=social_network.train_honest_nodes,
                                                                  sybil_targets=social_network.train_sybil_nodes)

                                    for algorithm in algorithms:
                                        if isinstance(algorithm, SybilGNN):
                                            algorithm.train_model = True
                                            algorithm.fine_tune = True

                                    # Evaluation
                                    evaluator = Evaluator(social_network=social_network,
                                                          label_noise_fraction=label_noise,
                                                          verbose=False)
                                    evaluator.evaluate_all(algorithms=algorithms,
                                                           reinitialize_GNNs=False)
                                    all_stats = evaluator.get_all_stats()
                                    for algo_id, algo in enumerate(algorithms):
                                        data_point = {
                                            "data_model_column": model,
                                            "graph_size_column": 2 * graph_size,
                                            "train_nodes_fraction_column": train_set_size,
                                            "label_noise_fraction_column": label_noise,
                                            "attack_edges_per_sybil": attack_edges_per_sybil,
                                            "algorithm": algo.name,
                                            "seed": seed
                                        }
                                        for stat_id, stat in enumerate(STATISTICS):
                                            experiment_data[
                                                model_id, graph_size_id, train_set_size_id, label_noise_id, att_id, algo_id, stat_id, i] = \
                                                all_stats[algo_id][stat]
                                            data_point[stat] = all_stats[algo_id][stat]
                                        raw_data.append(data_point)
                                        df_raw = pd.DataFrame(raw_data)
                                        write_experiment_csv(df_raw, directory, "raw_data.csv")

                                    # Progress bar
                                    p_bar.update(n=1)
                                    p_bar.refresh()

            experiment_mean = np.mean(experiment_data, axis=7)
            experiment_std = np.std(experiment_data, axis=7)

            for model_id, model in enumerate(models):
                for graph_size_id, graph_size in enumerate(graph_sizes):
                    for train_set_size_id, train_set_size in enumerate(train_set_sizes):
                        for label_noise_id, label_noise in enumerate(label_noises):
                            for att_id, attack_edges_per_sybil in enumerate(attack_edges):
                                for algo_id, algo in enumerate(algorithms):
                                    data_point = {
                                        "data_model_column": model,
                                        "graph_size_column": 2 * graph_size,
                                        "train_nodes_fraction_column": train_set_size,
                                        "label_noise_fraction_column": label_noise,
                                        "attack_edges_per_sybil": attack_edges_per_sybil,
                                        "algorithm": algo.name,
                                    }
                                    for stat_id, stat in enumerate(STATISTICS):
                                        data_point[f"{stat}_mean"] = experiment_mean[
                                            model_id, graph_size_id, train_set_size_id, label_noise_id, att_id, algo_id, stat_id]
                                        data_point[f"{stat}_std"] = experiment_std[
                                            model_id, graph_size_id, train_set_size_id, label_noise_id, att_id, algo_id, stat_id]
                                    data.append(data_point)

            df = pd.DataFrame(data)
            write_experiment_csv(df, directory, f"{variant}_data.csv")

            df_raw = pd.DataFrame(raw_data)
            write_experiment_csv(df_raw, directory, f"{variant}_raw_data.csv")
    if PLOT:
        print("Plotting thesis experiment 2")
        sns.set_theme()
        for var_id, variant in enumerate(variants):
            df_all = read_experiment_csv(directory, f"{variant}_data.csv")

            algo_list = ["\\textsc{SybilSCAR}",
                         "\\textsc{SybilGCN-L2}",
                         "\\textsc{SybilGCN-L4}",
                         "\\textsc{SybilRGCN-L2}",
                         "\\textsc{SybilGAT-L2}",
                         "\\textsc{SybilGAT-L4}"]
            df = df_all[df_all['algorithm'].isin(algo_list)]

            # plt.figure(figsize=(6, 3))
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Computer Modern Roman']
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

            plot = sns.relplot(
                data=df,
                x="attack_edges_per_sybil",
                y="AUC_mean",
                hue=str(variant),
                col="algorithm",
                kind="line",
                col_wrap=3,
                height=4,
                aspect=0.75
            )

            if variant == "data_model_column":
                legend_title = "Data Model"
                caption_1 = "Evaluation of AUC score on a network constructed with either the Barabási-Albert (BA) or Power Law (PL) model with 2000 nodes, with varying number of random attack edges per Sybil. Comparison of different data models for each algorithm."
                caption_2 = "Evaluation of AUC score on a network constructed with either the Barabási-Albert (BA) or Power Law (PL) model with 2000 nodes, with varying number of random attack edges per Sybil. Comparison of different algorithms for each data model."
            elif variant == "graph_size_column":
                legend_title = "Network Size"
                caption_1 = "Evaluation of AUC score on a network constructed with the Power Law (PL) model, with varying number of random attack edges per Sybil. Comparison of different network sizes (1000, 2000, 4000, 8000, 16'000) for each algorithm."
                caption_2 = "Evaluation of AUC score on a network constructed with the Power Law (PL) model, with varying number of random attack edges per Sybil. Comparison of different algorithms for each network size (1000, 2000, 4000, 8000, 16'000)."
            elif variant == "train_nodes_fraction_column":
                legend_title = "Train Nodes Fraction"
                caption_1 = "Evaluation of AUC score on a network constructed with the Power Law (PL) model with 2000 nodes, with varying number of random attack edges per Sybil. Comparison of different training nodes set fractions (0.5\%, 1\%, 5\%, 10\%, 20\%) for each algorithm."
                caption_2 = "Evaluation of AUC score on a network constructed with the Power Law (PL) model with 2000 nodes, with varying number of random attack edges per Sybil. Comparison of different algorithms for each training nodes set fraction (0.5\%, 1\%, 5\%, 10\%, 20\%)."
            elif variant == "label_noise_fraction_column":
                legend_title = "Label Noise Fraction"
                caption_1 = "Evaluation of AUC score on a network constructed with the Power Law (PL) model with 2000 nodes, with varying number of random attack edges per Sybil. Comparison of different label noise levels (0\%, 10\%, 20\%, 30\%) for each algorithm."
                caption_2 = "Evaluation of AUC score on a network constructed with the Power Law (PL) model with 2000 nodes, with varying number of random attack edges per Sybil. Comparison of different algorithms for each label noise level (0\%, 10\%, 20\%, 30\%)."
            else:
                raise Exception(f"Unknown variant: {variant}.")

            plot._legend.set_title(f"\\textbf{{{legend_title}}}")
            plot.set_titles(col_template="{col_name}")
            plot.set_xlabels("Attack edges per sybil")
            plot.set_ylabels("Mean AUC")

            sns.move_legend(plot, "center right", bbox_to_anchor=(1.0, 0.5))

            plt.ylim(0.5, 1)
            plt.xlim(2, 12)

            # plt.tight_layout()
            plt.savefig(f"{directory}/exp2_{variant}_by_algorithm.pdf")
            plt.savefig(f"{figures_directory}/exp2_{variant}_by_algorithm.pdf")
            plt.close()

            write_latex_figure_file(figures_directory,
                                    f"exp2_{variant}_by_algorithm.pdf",
                                    width=1.0,
                                    caption=caption_1,
                                    label=f"exp2_{variant}_by_algorithm",
                                    float_specifier="!htbp")

            # plt.figure(figsize=(6, 3))
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Computer Modern Roman']
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

            plot = sns.relplot(
                data=df,
                x="attack_edges_per_sybil",
                y="AUC_mean",
                hue="algorithm",
                col=str(variant),
                kind="line",
                col_wrap=3,
                palette=color_map,
                height=4,
                aspect=0.75
            )

            plot._legend.set_title("\\textbf{Algorithm}")
            plot.set_titles(col_template=f"{legend_title} = {{col_name}}")
            plot.set_xlabels("Attack edges per sybil")
            plot.set_ylabels("Mean AUC")

            sns.move_legend(plot, "center right", bbox_to_anchor=(1.01, 0.5))

            plt.ylim(0.5, 1)
            plt.xlim(2, 12)

            # plt.tight_layout()
            plt.savefig(f"{directory}/exp2_{variant}_by_category.pdf")
            plt.savefig(f"{figures_directory}/exp2_{variant}_by_category.pdf")
            plt.close()

            write_latex_figure_file(figures_directory,
                                    f"exp2_{variant}_by_category.pdf",
                                    width=1.0,
                                    caption=caption_2,
                                    label=f"exp2_{variant}_by_category",
                                    float_specifier="!htbp")

if EXPERIMENT_3:
    directory = "thesis_experiments/experiment_3"
    if RUN:
        print("Running thesis experiment 3")
        data = []
        raw_data = []

        total_num_experiments = NUM_EXPERIMENTS
        p_bar = tqdm(total=total_num_experiments)

        experiment_data = np.zeros(shape=(NUM_ALGORITHMS, len(STATISTICS), NUM_EXPERIMENTS))
        algorithms = None
        for i in range(NUM_EXPERIMENTS):
            seed = int(SEEDS[i])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            algorithms = combine_and_copy_lists(baseline_algorithms, gnn_algorithms)

            social_network = Twitter270K(is_directed=False)
            number_of_nodes = social_network.network.num_nodes() // 20

            # Network sampling
            subsampled_social_network, remaining_social_network = ForestFireSampler(
                seed=seed).sample_social_network(
                social_network=social_network,
                number_of_nodes=number_of_nodes,
                num_sources=1,
                verbose=False)

            pretrain_algorithms = []
            for algorithm in algorithms:
                if isinstance(algorithm, SybilGNN):
                    # Put algorithm into training mode
                    algorithm.train_model = True
                    algorithm.fine_tune = False
                    pretrain_algorithms.append(algorithm)

            pretrain_evaluator = Evaluator(social_network=subsampled_social_network,
                                           verbose=False)
            pretrain_evaluator.evaluate_all(algorithms=pretrain_algorithms)

            # for algorithm in pretrain_algorithms:
            #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
            #                     f"{directory}/training_curves/{algorithm}_exp{i}_pretrain.pdf")

            # pretrain_evaluator.get_all_stats()

            for algorithm in algorithms:
                if isinstance(algorithm, SybilGNN):
                    algorithm.pretrain_runtime = algorithm.runtime
                    # Do not fine-tune, directly apply to remaining social network
                    # algorithm.train_model = False
                    # Fine-tune
                    algorithm.fine_tune = True
                    algorithm.num_epochs = FINE_TUNE_EPOCHS
                    algorithm.patience = 3

            # Evaluation on Remaining Network
            evaluator = Evaluator(social_network=remaining_social_network,
                                  verbose=False)
            evaluator.evaluate_all(algorithms=algorithms,
                                   reinitialize_GNNs=False)

            # for algorithm in pretrain_algorithms:
            #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
            #                     f"{directory}/training_curves/{algorithm}_exp{i}_eval.pdf")

            all_stats = evaluator.get_all_stats()
            for algo_id, algo in enumerate(algorithms):
                data_point = {
                    "algorithm": algo.name,
                    "seed": seed
                }
                for stat_id, stat in enumerate(STATISTICS):
                    experiment_data[algo_id, stat_id, i] = all_stats[algo_id][stat]
                    data_point[stat] = all_stats[algo_id][stat]
                raw_data.append(data_point)
                df_raw = pd.DataFrame(raw_data)
                write_experiment_csv(df_raw, directory, "raw_data.csv")

            # Progress bar
            p_bar.update(n=1)
            p_bar.refresh()

        experiment_mean = np.mean(experiment_data, axis=2)
        experiment_std = np.std(experiment_data, axis=2)

        for algo_id, algo in enumerate(algorithms):
            data_point = {
                "algorithm": algo.name
            }
            for stat_id, stat in enumerate(STATISTICS):
                data_point[f"{stat}_mean"] = experiment_mean[algo_id, stat_id]
                data_point[f"{stat}_std"] = experiment_std[algo_id, stat_id]
            data.append(data_point)

        df = pd.DataFrame(data)
        write_experiment_csv(df, directory)

        df_raw = pd.DataFrame(raw_data)
        write_experiment_csv(df_raw, directory, "raw_data.csv")
    if PLOT:
        print("Plotting thesis experiment 3")
        sns.set_theme()

        df_raw = read_experiment_csv(directory, "raw_data.csv")

        plt.figure(figsize=(10, 4))
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        ax = sns.barplot(
            y="AUC",
            hue="algorithm",
            data=df_raw,
            capsize=0.1,
            err_kws={'linewidth': 1, 'color': 'black'},
            errorbar="sd",
            palette=color_map
        )

        legend = plt.legend(title="\\textbf{Algorithm}", bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_edgecolor('none')

        plt.ylim(0.4, 1)

        plt.ylabel("AUC")
        plt.xlabel(None)

        plt.tight_layout()
        plt.savefig(f"{directory}/exp3_twitter.pdf")
        plt.savefig(f"{figures_directory}/exp3_twitter.pdf")
        plt.close()

        write_latex_figure_file(figures_directory,
                                "exp3_twitter.pdf",
                                width=1.0,
                                caption="Evaluation of AUC score on Twitter dataset, after pre-training on a sampled subgraph 5\% of the size of the original network. Evaluation performed on the remaining network.",
                                label="exp3_twitter")
    if TABLES:
        print("Making tables for thesis experiment 3")
        df = read_experiment_csv(directory)
        df = reformat_dataframe(df)

        latex_table = df.to_latex(
            columns=["algorithm", "AUC", "accuracy", "precision", "recall", "pretrain_runtime",
                     "runtime"],
            header=["Algorithm", "AUC", "Accuracy", "Precision", "Recall", "Pretrain Runtime",
                    "Runtime"],
            index=False,
            float_format="{:.3f}".format
        )
        latex_table = latex_table.replace("nan ± nan", "")
        latex_table = latex_table.replace("\\textbackslash textsc", "\\textsc")
        latex_table = latex_table.replace("\\{", "{")
        latex_table = latex_table.replace("\\}", "}")

        latex_table_file = r"""
                        \begin{table}
                        \centering
                        \caption{Complete data for the experiments from \Cref{sec:twitter-dataset-evaluation}. Runtimes are in seconds. Five runs for each experiment.}
                        \label{tab:exp3_full_data}
                        \small % or \footnotesize for even smaller text
                        \begin{adjustbox}{width=\textwidth}
                        """ + latex_table + r"""
                        \end{adjustbox}
                        \end{table}
                        """

        with open('thesis_experiments/thesis_tables/exp3_full_data.tex', 'w') as f:
            f.write(latex_table_file)

if EXPERIMENT_4_1:
    directory = "thesis_experiments/experiment_4_1"
    if RUN:
        print("Running thesis experiment 4.1")
        data = []
        raw_data = []

        attack_edges = [2, 4, 6, 8, 10, 12, 14, 16]

        attacks = [
            Attack(p_targeted=0.05, pdf_targeted=[0.25, 0.25, 0.5]),
            Attack(p_targeted=0.1, pdf_targeted=[0.25, 0.25, 0.5]),
            Attack(p_targeted=0.15, pdf_targeted=[0.25, 0.25, 0.5]),
            Attack(p_targeted=0.2, pdf_targeted=[0.25, 0.25, 0.5]),
        ]
        algorithms = None

        experiment_data = np.zeros(shape=(len(attacks),
                                          len(attack_edges),
                                          NUM_ALGORITHMS,
                                          len(STATISTICS),
                                          NUM_EXPERIMENTS)
                                   )

        total_counter = len(attacks) * len(attack_edges) * NUM_EXPERIMENTS
        p_bar = tqdm(total=total_counter)

        for attack_id, attack in enumerate(attacks):
            for att_id, attack_edges_per_sybil in enumerate(attack_edges):
                for i in range(NUM_EXPERIMENTS):
                    seed = int(SEEDS[i])
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    algorithms = combine_and_copy_lists(baseline_algorithms, gnn_algorithms)

                    social_network = SocialNetworkFromRegions(honest_region=PowerLawGraph(1000, 10, 0.8),
                                                              sybil_region=PowerLawGraph(1000, 10, 0.8, is_sybil=True))
                    social_network.train_test_split(train_fraction=0.05)
                    attack.perform_attack(social_network=social_network,
                                          num_attack_edges=attack_edges_per_sybil * social_network.sybil_region.num_nodes(),
                                          honest_targets=social_network.train_honest_nodes,
                                          sybil_targets=social_network.train_sybil_nodes)

                    for algorithm in algorithms:
                        if isinstance(algorithm, SybilGNN):
                            algorithm.train_model = True
                            algorithm.fine_tune = True

                    # Evaluation
                    evaluator = Evaluator(social_network=social_network,
                                          verbose=False)
                    evaluator.evaluate_all(algorithms=algorithms,
                                           reinitialize_GNNs=False)
                    all_stats = evaluator.get_all_stats()
                    for algo_id, algo in enumerate(algorithms):
                        data_point = {
                            "attack": str(attack),
                            "attack_edges_per_sybil": attack_edges_per_sybil,
                            "algorithm": algo.name,
                            "seed": seed
                        }
                        for stat_id, stat in enumerate(STATISTICS):
                            experiment_data[attack_id, att_id, algo_id, stat_id, i] = all_stats[algo_id][stat]
                            data_point[stat] = all_stats[algo_id][stat]
                        raw_data.append(data_point)
                        df_raw = pd.DataFrame(raw_data)
                        write_experiment_csv(df_raw, directory, "raw_data.csv")

                    # Progress bar
                    p_bar.update(n=1)
                    p_bar.refresh()

        experiment_mean = np.mean(experiment_data, axis=4)
        experiment_std = np.std(experiment_data, axis=4)

        for attack_id, attack in enumerate(attacks):
            for att_id, attack_edges_per_sybil in enumerate(attack_edges):
                for algo_id, algo in enumerate(algorithms):
                    data_point = {
                        "attack": str(attack),
                        "attack_edges_per_sybil": attack_edges_per_sybil,
                        "algorithm": algo.name,
                    }
                    for stat_id, stat in enumerate(STATISTICS):
                        data_point[f"{stat}_mean"] = experiment_mean[attack_id, att_id, algo_id, stat_id]
                        data_point[f"{stat}_std"] = experiment_std[attack_id, att_id, algo_id, stat_id]
                    data.append(data_point)

        df = pd.DataFrame(data)
        write_experiment_csv(df, directory)

        df_raw = pd.DataFrame(raw_data)
        write_experiment_csv(df_raw, directory, "raw_data.csv")
    if PLOT:
        print("Plotting thesis experiment 4.1")
        df_all = read_experiment_csv(directory, make_attack_replacements=True)

        algo_list = ["\\textsc{SybilRank}",
                     "\\textsc{SybilBelief}",
                     "\\textsc{SybilSCAR}",
                     "\\textsc{SybilGCN-L2}",
                     "\\textsc{SybilGCN-L4}",
                     "\\textsc{SybilRGCN-L2}",
                     "\\textsc{SybilGAT-L2}",
                     "\\textsc{SybilGAT-L4}"]
        df = df_all[df_all['algorithm'].isin(algo_list)]

        sns.set_theme()

        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        plot = sns.relplot(
            data=df,
            x="attack_edges_per_sybil",
            y="AUC_mean",
            hue="attack",
            col="algorithm",
            kind="line",
            col_wrap=3,
            height=4,
            aspect=0.75
        )
        plot._legend.set_title("\\textbf{Attack}")
        plot.set_titles(col_template="{col_name}")
        plot.set_xlabels("Attack edges per sybil")
        plot.set_ylabels("Mean AUC")

        sns.move_legend(plot, "center right", bbox_to_anchor=(1.01, 0.5))

        plt.ylim(0.4, 1)
        plt.xlim(2, 12)

        plt.savefig(f"{directory}/exp4_1_by_algorithm.pdf")
        plt.savefig(f"{figures_directory}/exp4_1_by_algorithm.pdf")
        plt.close()

        write_latex_figure_file(figures_directory,
                                "exp4_1_by_algorithm.pdf",
                                width=1.0,
                                caption="Evaluation of AUC score on a network constructed with the Power Law (PL) model with 2000 nodes, with varying number of \emph{targeted} attack edges per Sybil. Comparison of different attacks for each algorithm.",
                                label="exp4_1_by_algorithm")

        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        plot = sns.relplot(
            data=df,
            x="attack_edges_per_sybil",
            y="AUC_mean",
            hue="algorithm",
            col="attack",
            kind="line",
            palette=color_map,
            col_wrap=2,
            height=4,
            aspect=0.75
        )
        plot._legend.set_title("\\textbf{Algorithm}")
        plot.set_titles(col_template="{col_name}")
        plot.set_xlabels("Attack edges per sybil")
        plot.set_ylabels("Mean AUC")

        sns.move_legend(plot, "center right", bbox_to_anchor=(1.01, 0.5))

        plt.ylim(0.4, 1)
        plt.xlim(2, 12)

        plt.savefig(f"{directory}/exp4_1_by_attack.pdf")
        plt.savefig(f"{figures_directory}/exp4_1_by_attack.pdf")
        plt.close()

        write_latex_figure_file(figures_directory,
                                "exp4_1_by_attack.pdf",
                                width=0.8,
                                caption="Evaluation of AUC score on a network constructed with the Power Law (PL) model with 2000 nodes, with varying number of \emph{targeted} attack edges per Sybil. Comparison of different algorithms for each attack.",
                                label="exp4_1_by_attack")

if EXPERIMENT_4_2:
    variants = ["BA", "PL"]
    directory = "thesis_experiments/experiment_4_2"
    if RUN:
        print("Running thesis experiment 1.2")
        data = []
        raw_data = []

        total_counter = len(variants) * NUM_EXPERIMENTS
        p_bar = tqdm(total=total_counter)

        experiment_data = np.zeros(shape=(len(variants), NUM_ALGORITHMS,
                                          len(STATISTICS), NUM_EXPERIMENTS))
        algorithms = None
        for var_id, variant in enumerate(variants):
            for i in range(NUM_EXPERIMENTS):
                seed = int(SEEDS[i])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                algorithms = combine_and_copy_lists(baseline_algorithms, gnn_algorithms)

                graph_size = 1000

                attack_edges_per_sybil = 10
                if variant == "BA":
                    pretrain_honest_region = BarabasiAlbertGraph(graph_size, 10)
                    pretrain_sybil_region = BarabasiAlbertGraph(graph_size, 10, is_sybil=True)
                    eval_honest_region = BarabasiAlbertGraph(graph_size, 10)
                    eval_sybil_region = BarabasiAlbertGraph(graph_size, 10, is_sybil=True)
                elif variant == "PL":
                    pretrain_honest_region = PowerLawGraph(graph_size, 10, 0.8)
                    pretrain_sybil_region = PowerLawGraph(graph_size, 10, 0.8, is_sybil=True)
                    eval_honest_region = PowerLawGraph(graph_size, 10, 0.8)
                    eval_sybil_region = PowerLawGraph(graph_size, 10, 0.8, is_sybil=True)
                else:
                    raise Exception("Invalid variant")

                pretrain_social_network = SocialNetworkFromRegions(honest_region=pretrain_honest_region,
                                                                   sybil_region=pretrain_sybil_region)
                pretrain_social_network.train_test_split(train_fraction=0.05)
                RandomAttack().perform_attack(social_network=pretrain_social_network,
                                              num_attack_edges=attack_edges_per_sybil * pretrain_social_network.sybil_region.num_nodes(),
                                              honest_targets=pretrain_social_network.train_honest_nodes,
                                              sybil_targets=pretrain_social_network.train_sybil_nodes)

                pretrain_algorithms = []
                for algorithm in algorithms:
                    if isinstance(algorithm, SybilGNN):
                        # Put algorithm into training mode
                        algorithm.train_model = True
                        algorithm.fine_tune = False
                        pretrain_algorithms.append(algorithm)

                pretrain_evaluator = Evaluator(social_network=pretrain_social_network,
                                               verbose=False)
                pretrain_evaluator.evaluate_all(algorithms=pretrain_algorithms)

                # for algorithm in pretrain_algorithms:
                #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
                #                     f"{directory}/training_curves/{variant}_{algorithm}_exp{i}_pretrain.pdf")

                # pretrain_evaluator.get_all_stats()

                for algorithm in algorithms:
                    if isinstance(algorithm, SybilGNN):
                        algorithm.pretrain_runtime = algorithm.runtime
                        # Do not fine-tune, directly apply to remaining social network
                        # algorithm.train_model = False
                        # Fine-tune
                        algorithm.fine_tune = True
                        algorithm.num_epochs = FINE_TUNE_EPOCHS
                        algorithm.patience = 3

                eval_social_network = SocialNetworkFromRegions(honest_region=eval_honest_region,
                                                               sybil_region=eval_sybil_region)
                eval_social_network.train_test_split(train_fraction=0.05)
                Attack(p_targeted=0.05, pdf_targeted=[0.25, 0.25, 0.5]).perform_attack(
                    social_network=eval_social_network,
                    num_attack_edges=attack_edges_per_sybil * eval_social_network.sybil_region.num_nodes(),
                    honest_targets=eval_social_network.train_honest_nodes,
                    sybil_targets=eval_social_network.train_sybil_nodes)

                # Evaluation
                evaluator = Evaluator(social_network=eval_social_network,
                                      verbose=False)
                evaluator.evaluate_all(algorithms=algorithms,
                                       reinitialize_GNNs=False)

                # for algorithm in pretrain_algorithms:
                #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
                #                     f"{directory}/training_curves/{variant}_{algorithm}_exp{i}_eval.pdf")

                all_stats = evaluator.get_all_stats()
                for algo_id, algo in enumerate(algorithms):
                    data_point = {
                        "variant": variant,
                        "algorithm": algo.name,
                        "seed": seed
                    }
                    for stat_id, stat in enumerate(STATISTICS):
                        experiment_data[var_id, algo_id, stat_id, i] = all_stats[algo_id][stat]
                        data_point[stat] = all_stats[algo_id][stat]
                    raw_data.append(data_point)
                    df_raw = pd.DataFrame(raw_data)
                    write_experiment_csv(df_raw, directory, "raw_data.csv")

                # Progress bar
                p_bar.update(n=1)
                p_bar.refresh()

        experiment_mean = np.mean(experiment_data, axis=3)
        experiment_std = np.std(experiment_data, axis=3)

        for var_id, variant in enumerate(variants):
            for algo_id, algo in enumerate(algorithms):
                data_point = {
                    "variant": variant,
                    "algorithm": algo.name
                }
                for stat_id, stat in enumerate(STATISTICS):
                    data_point[f"{stat}_mean"] = experiment_mean[var_id, algo_id, stat_id]
                    data_point[f"{stat}_std"] = experiment_std[var_id, algo_id, stat_id]
                data.append(data_point)

        df = pd.DataFrame(data)
        write_experiment_csv(df, directory)

        df_raw = pd.DataFrame(raw_data)
        write_experiment_csv(df_raw, directory, "raw_data.csv")
    if PLOT:
        print("Plotting thesis experiment 4.2")
        sns.set_theme()
        df_raw = read_experiment_csv(directory, "raw_data.csv")

        plt.figure(figsize=(10, 4))
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        ax = sns.barplot(
            x="variant",
            y="AUC",
            hue="algorithm",
            data=df_raw,
            capsize=0.1,
            err_kws={'linewidth': 1, 'color': 'black'},
            errorbar="sd",
            palette=color_map
        )

        legend = plt.legend(title="\\textbf{Algorithm}", bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_edgecolor('none')

        plt.ylim(0.3, None)

        plt.ylabel("AUC")
        plt.xlabel(None)

        plt.tight_layout()
        plt.savefig(f"{directory}/exp4_2.pdf")
        plt.savefig(f"{figures_directory}/exp4_2.pdf")
        plt.close()

        write_latex_figure_file(figures_directory,
                                "exp4_2.pdf",
                                width=1.0,
                                caption="Evaluation of AUC score on a network constructed with the Power Law (PL) model with 2000 nodes, which was attacked with 10 targeted attack edges per Sybil, after being pre-trained on a the same network constructed with 10 \emph{random} attack edges per Sybil.",
                                label="exp4_2")
    if TABLES:
        print("Making tables for thesis experiment 4.2")
        df = read_experiment_csv(directory)
        df = reformat_dataframe(df)

        latex_table = df.to_latex(
            columns=["variant", "algorithm", "AUC", "accuracy", "precision", "recall", "pretrain_runtime",
                     "runtime"],
            header=["Variant", "Algorithm", "AUC", "Accuracy", "Precision", "Recall", "Pretrain Runtime",
                    "Runtime"],
            index=False,
            float_format="{:.3f}".format
        )
        latex_table = latex_table.replace("nan ± nan", "")
        latex_table = latex_table.replace("\\textbackslash textsc", "\\textsc")
        latex_table = latex_table.replace("\\{", "{")
        latex_table = latex_table.replace("\\}", "}")

        latex_table_file = r"""
                        \begin{table}
                        \centering
                        \caption{Complete data for the experiments from \Cref{sec:pretraining-before-attack}. Runtimes are in seconds. Five runs for each experiment.}
                        \label{tab:exp4_2_full_data}
                        \small % or \footnotesize for even smaller text
                        \begin{adjustbox}{width=\textwidth}
                        """ + latex_table + r"""
                        \end{adjustbox}
                        \end{table}
                        """

        with open('thesis_experiments/thesis_tables/exp4_2_full_data.tex', 'w') as f:
            f.write(latex_table_file)

if EXPERIMENT_5_1:
    directory = "thesis_experiments/experiment_5_1"
    if RUN:
        print("Running thesis experiment 5.1")
        data = []
        raw_data = []

        graph_sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000]

        total_counter = len(graph_sizes) * NUM_EXPERIMENTS
        p_bar = tqdm(total=total_counter)

        experiment_data = np.zeros(shape=(len(graph_sizes), NUM_ALGORITHMS, len(STATISTICS), NUM_EXPERIMENTS))
        algorithms = None
        for graph_size_id, graph_size in enumerate(graph_sizes):
            for i in range(NUM_EXPERIMENTS):
                seed = int(SEEDS[i])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                algorithms = combine_and_copy_lists(baseline_algorithms, gnn_algorithms)

                pretrain_size = 1000
                eval_size = graph_size

                pretrain_attack_edges_per_sybil = 8
                eval_attack_edges_per_sybil = 8

                pretrain_social_network = SocialNetworkFromRegions(
                    honest_region=PowerLawGraph(pretrain_size, 6, 0.8),
                    sybil_region=PowerLawGraph(pretrain_size, 6, 0.8,
                                               is_sybil=True))
                eval_social_network = SocialNetworkFromRegions(
                    honest_region=PowerLawGraph(eval_size, 6, 0.8),
                    sybil_region=PowerLawGraph(eval_size, 6, 0.8,
                                               is_sybil=True))

                pretrain_social_network.train_test_split(train_fraction=0.05)
                eval_social_network.train_test_split(train_fraction=0.05)

                RandomAttack().perform_attack(social_network=pretrain_social_network,
                                              num_attack_edges=pretrain_attack_edges_per_sybil * pretrain_social_network.sybil_region.num_nodes(),
                                              honest_targets=pretrain_social_network.train_honest_nodes,
                                              sybil_targets=pretrain_social_network.train_sybil_nodes)

                RandomAttack().perform_attack(social_network=eval_social_network,
                                              num_attack_edges=eval_attack_edges_per_sybil * eval_social_network.sybil_region.num_nodes(),
                                              honest_targets=eval_social_network.train_honest_nodes,
                                              sybil_targets=eval_social_network.train_sybil_nodes)

                pretrain_algorithms = []
                for algorithm in algorithms:
                    if isinstance(algorithm, SybilGNN):
                        # Put algorithm into training mode
                        algorithm.train_model = True
                        algorithm.fine_tune = False
                        pretrain_algorithms.append(algorithm)

                pretrain_evaluator = Evaluator(social_network=pretrain_social_network,
                                               verbose=False)
                pretrain_evaluator.evaluate_all(algorithms=pretrain_algorithms)

                # for algorithm in pretrain_algorithms:
                #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
                #                     f"{directory}/training_curves/size{graph_size}_{algorithm}_exp{i}_pretrain.pdf")

                # pretrain_evaluator.get_all_stats()

                for algorithm in algorithms:
                    if isinstance(algorithm, SybilGNN):
                        algorithm.pretrain_runtime = algorithm.runtime
                        # Do not fine-tune, directly apply to remaining social network
                        # algorithm.train_model = False
                        # Fine-tune
                        algorithm.fine_tune = True
                        algorithm.num_epochs = FINE_TUNE_EPOCHS
                        algorithm.patience = 3

                # Evaluation
                evaluator = Evaluator(social_network=eval_social_network,
                                      verbose=False)
                evaluator.evaluate_all(algorithms=algorithms,
                                       reinitialize_GNNs=False)

                # for algorithm in pretrain_algorithms:
                #    plot_loss_curves(algorithm.train_losses, algorithm.val_losses,
                #                     f"{directory}/training_curves/size{graph_size}_{algorithm}_exp{i}_eval.pdf")

                all_stats = evaluator.get_all_stats()
                for algo_id, algo in enumerate(algorithms):
                    data_point = {
                        "graph_size": 2 * graph_size,
                        "algorithm": algo.name,
                        "seed": seed
                    }
                    for stat_id, stat in enumerate(STATISTICS):
                        experiment_data[graph_size_id, algo_id, stat_id, i] = all_stats[algo_id][stat]
                        data_point[stat] = all_stats[algo_id][stat]
                    raw_data.append(data_point)
                    df_raw = pd.DataFrame(raw_data)
                    write_experiment_csv(df_raw, directory, "raw_data.csv")

                # Progress bar
                p_bar.update(n=1)
                p_bar.refresh()

        experiment_mean = np.mean(experiment_data, axis=3)
        experiment_std = np.std(experiment_data, axis=3)

        for graph_size_id, graph_size in enumerate(graph_sizes):
            for algo_id, algo in enumerate(algorithms):
                data_point = {
                    "graph_size": 2 * graph_size,
                    "algorithm": algo.name
                }
                for stat_id, stat in enumerate(STATISTICS):
                    data_point[f"{stat}_mean"] = experiment_mean[graph_size_id, algo_id, stat_id]
                    data_point[f"{stat}_std"] = experiment_std[graph_size_id, algo_id, stat_id]
                data.append(data_point)

        df = pd.DataFrame(data)
        write_experiment_csv(df, directory)

        df_raw = pd.DataFrame(raw_data)
        write_experiment_csv(df_raw, directory, "raw_data.csv")
    if PLOT:
        print("Plotting thesis experiment 5.1")
        sns.set_theme()

        df_all = read_experiment_csv(directory, "raw_data.csv")

        df_raw = df_all[df_all["graph_size"].isin([2000, 8000, 32000, 128000])]

        plt.figure(figsize=(10, 4))
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        sns.barplot(data=df_raw,
                    x='graph_size',
                    y='AUC',
                    hue='algorithm',
                    capsize=0.1,
                    err_kws={'linewidth': 1, 'color': 'black'},
                    errorbar="sd",
                    palette=color_map)

        legend = plt.legend(title="\\textbf{Algorithm}", bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_edgecolor('none')

        plt.ylim(0.4, 1)

        plt.xlabel('Social Network Size')
        plt.ylabel('AUC')

        plt.tight_layout()
        plt.savefig(f"{directory}/exp5_1.pdf")
        plt.savefig(f"{figures_directory}/exp5_1.pdf")
        plt.close()

        write_latex_figure_file(figures_directory,
                                f"exp5_1.pdf",
                                width=1.0,
                                caption="Evaluation of AUC score on networks of exponentially growing size, after pre-training on a small network of size 2000 nodes. The networks were constructed using the Power Law (PL) model.",
                                label=f"exp5_1")
    if TABLES:
        print("Making tables for thesis experiment 5.1")
        df = read_experiment_csv(directory)
        df = reformat_dataframe(df)

        latex_table = df.to_latex(
            columns=["graph_size", "algorithm", "AUC", "accuracy", "precision", "recall", "pretrain_runtime",
                     "runtime"],
            header=["Graph Size", "Algorithm", "AUC", "Accuracy", "Precision", "Recall", "Pretrain Runtime",
                    "Runtime"],
            index=False,
            float_format="{:.3f}".format
        )
        latex_table = latex_table.replace("nan ± nan", "")
        latex_table = latex_table.replace("\\textbackslash textsc", "\\textsc")
        latex_table = latex_table.replace("\\{", "{")
        latex_table = latex_table.replace("\\}", "}")

        latex_table_file = r"""
                                \begin{table}
                                \centering
                                \caption{Complete data for the experiments from \Cref{sec:experiment-scaling-to-very-large-networks}. Runtimes are in seconds. Five runs for each experiment.}
                                \label{tab:exp5_1_full_data}
                                \small % or \footnotesize for even smaller text
                                \begin{adjustbox}{width=\textwidth}
                                """ + latex_table + r"""
                                \end{adjustbox}
                                \end{table}
                                """

        with open('thesis_experiments/thesis_tables/exp5_1_full_data.tex', 'w') as f:
            f.write(latex_table_file)

if EXPERIMENT_5_2:
    directory = "thesis_experiments/experiment_5_2"
    if RUN:
        print("Running thesis experiment 5.2")
        data = []
        raw_data = []

        base_size = 1000
        base_regions = ["honest", "sybil"]
        multipliers = [2, 4, 6]

        experiment_data = np.zeros(shape=(len(base_regions),
                                          len(multipliers),
                                          NUM_ALGORITHMS,
                                          len(STATISTICS),
                                          NUM_EXPERIMENTS)
                                   )

        total_counter = len(base_regions) * len(multipliers) * NUM_EXPERIMENTS
        p_bar = tqdm(total=total_counter)
        algorithms = None
        for base_reg_id, base_region in enumerate(base_regions):
            for mult_id, multiplier in enumerate(multipliers):
                for i in range(NUM_EXPERIMENTS):
                    seed = int(SEEDS[i])
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    algorithms = combine_and_copy_lists(baseline_algorithms, gnn_algorithms)

                    multiplier_size = multiplier * base_size
                    attack_edges_per_sybil = 8
                    if base_region == "honest":
                        honest_region = PowerLawGraph(base_size, 6, 0.8)
                        sybil_region = PowerLawGraph(multiplier_size, 6, 0.8, is_sybil=True)
                    elif base_region == "sybil":
                        honest_region = PowerLawGraph(multiplier_size, 6, 0.8)
                        sybil_region = PowerLawGraph(base_size, 6, 0.8, is_sybil=True)
                    else:
                        raise Exception()

                    social_network = SocialNetworkFromRegions(honest_region=honest_region,
                                                              sybil_region=sybil_region)
                    social_network.train_test_split(train_fraction=0.05)
                    total_nodes = base_size + multiplier_size
                    RandomAttack().perform_attack(social_network=social_network,
                                                  num_attack_edges=attack_edges_per_sybil * (total_nodes // 2),
                                                  honest_targets=social_network.train_honest_nodes,
                                                  sybil_targets=social_network.train_sybil_nodes)

                    for algorithm in algorithms:
                        if isinstance(algorithm, SybilGNN):
                            algorithm.train_model = True
                            algorithm.fine_tune = True

                    # Evaluation
                    evaluator = Evaluator(social_network=social_network,
                                          verbose=False)
                    evaluator.evaluate_all(algorithms=algorithms,
                                           reinitialize_GNNs=False)
                    all_stats = evaluator.get_all_stats()

                    honest_size = base_size if base_region == "honest" else base_size * multiplier
                    sybil_size = base_size if base_region == "sybil" else base_size * multiplier
                    for algo_id, algo in enumerate(algorithms):
                        data_point = {
                            "honest_size": honest_size,
                            "sybil_size": sybil_size,
                            "algorithm": algo.name,
                            "seed": seed
                        }
                        for stat_id, stat in enumerate(STATISTICS):
                            experiment_data[base_reg_id, mult_id, algo_id, stat_id, i] = all_stats[algo_id][stat]
                            data_point[stat] = all_stats[algo_id][stat]
                        raw_data.append(data_point)
                        df_raw = pd.DataFrame(raw_data)
                        write_experiment_csv(df_raw, directory, "raw_data.csv")

                    # Progress bar
                    p_bar.update(n=1)
                    p_bar.refresh()

        experiment_mean = np.mean(experiment_data, axis=4)
        experiment_std = np.std(experiment_data, axis=4)

        for base_reg_id, base_region in enumerate(base_regions):
            for mult_id, multiplier in enumerate(multipliers):
                honest_size = base_size if base_region == "honest" else base_size * multiplier
                sybil_size = base_size if base_region == "sybil" else base_size * multiplier
                for algo_id, algo in enumerate(algorithms):
                    data_point = {
                        "honest_size": honest_size,
                        "sybil_size": sybil_size,
                        "algorithm": algo.name,
                    }
                    for stat_id, stat in enumerate(STATISTICS):
                        data_point[f"{stat}_mean"] = experiment_mean[base_reg_id, mult_id, algo_id, stat_id]
                        data_point[f"{stat}_std"] = experiment_std[base_reg_id, mult_id, algo_id, stat_id]
                    data.append(data_point)

        df = pd.DataFrame(data)
        write_experiment_csv(df, directory)

        df_raw = pd.DataFrame(raw_data)
        write_experiment_csv(df_raw, directory, "raw_data.csv")
    if PLOT:
        print("Plotting thesis experiment 5.2")
        sns.set_theme()

        df_raw = read_experiment_csv(directory, "raw_data.csv")
        for base_reg in ["honest_size", "sybil_size"]:
            df_raw_base_reg = df_raw[df_raw[base_reg] == 1000]
            x = "sybil_size" if base_reg == "honest_size" else "honest_size"

            plt.figure(figsize=(10, 4))
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Computer Modern Roman']
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

            sns.barplot(data=df_raw_base_reg,
                        x=x,
                        y="AUC",
                        hue="algorithm",
                        capsize=0.1,
                        err_kws={'linewidth': 1, 'color': 'black'},
                        errorbar="sd",
                        palette=color_map)

            legend = plt.legend(title="\\textbf{Algorithm}", bbox_to_anchor=(1.05, 1), loc='upper left')
            legend.get_frame().set_facecolor('none')
            legend.get_frame().set_edgecolor('none')

            plt.ylim(0, 1)
            plt.ylabel("AUC")
            x_label = "Sybil region size" if base_reg == "honest_size" else "Honest region size"
            plt.xlabel(x_label)

            title = "Honest region size = 1000 nodes" if base_reg == "honest_size" else "Sybil region size = 1000 nodes"
            plt.title(title)

            plt.tight_layout()
            plt.savefig(f"{directory}/exp5_2_base_{base_reg}.pdf")
            plt.savefig(f"{figures_directory}/exp5_2_base_{base_reg}.pdf")
            plt.close()

            if base_reg == "honest_size":
                caption = "Evaluation of AUC score on a network that was constructed with the Power Law (PL) model with 1000 nodes in the honest region, and varying number of nodes in the Sybil region."
            else:
                caption = "Evaluation of AUC score on a network that was constructed with the Power Law (PL) model with 1000 nodes in the Sybil region, and varying number of nodes in the honest region."

            write_latex_figure_file(figures_directory,
                                    f"exp5_2_base_{base_reg}.pdf",
                                    width=1.0,
                                    caption=caption,
                                    label=f"exp5_2_base_{base_reg}")
    if TABLES:
        print("Making tables for thesis experiment 5.2")
        df = read_experiment_csv(directory)
        df = reformat_dataframe(df)

        latex_table = df.to_latex(
            columns=["honest_size", "sybil_size", "algorithm", "AUC", "accuracy", "precision", "recall",
                     "pretrain_runtime",
                     "runtime"],
            header=["Honest Size", "Sybil Size", "Algorithm", "AUC", "Accuracy", "Precision", "Recall",
                    "Pretrain Runtime",
                    "Runtime"],
            index=False,
            float_format="{:.3f}".format
        )
        latex_table = latex_table.replace("nan ± nan", "")
        latex_table = latex_table.replace("\\textbackslash textsc", "\\textsc")
        latex_table = latex_table.replace("\\{", "{")
        latex_table = latex_table.replace("\\}", "}")

        latex_table_file = r"""
                                        \begin{table}
                                        \centering
                                        \caption{Complete data for the experiments from \Cref{sec:experiment-different-region-sizes}. Runtimes are in seconds. Five runs for each experiment.}
                                        \label{tab:exp5_2_full_data}
                                        \small % or \footnotesize for even smaller text
                                        \begin{adjustbox}{width=\textwidth}
                                        """ + latex_table + r"""
                                        \end{adjustbox}
                                        \end{table}
                                        """

        with open('thesis_experiments/thesis_tables/exp5_2_full_data.tex', 'w') as f:
            f.write(latex_table_file)

if EXPERIMENT_5_3:
    source_directory = "thesis_experiments/experiment_2"
    directory = "thesis_experiments/experiment_5_3"
    if PLOT:
        print("Plotting thesis experiment 5.3")
        sns.set_theme()

        accuracy_experiments = [
            ("1_1", "variant", "_STAT_ scores of experiments from \Cref{sec:pretraining-sampled-subgraph}."),
            ("1_2", "variant", "_STAT_ scores of experiments from \Cref{sec:pretraining-smaller-network}."),
        ]
        for stat in ["accuracy", "precision", "recall", "F1"]:
            for (runtime_experiment, x, caption) in accuracy_experiments:
                df_raw = read_experiment_csv(f"thesis_experiments/experiment_{runtime_experiment}", "raw_data.csv")

                plt.figure(figsize=(10, 4))
                plt.rcParams['text.usetex'] = True
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = ['Computer Modern Roman']
                plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

                plot = sns.barplot(data=df_raw,
                                   x=x,
                                   y=stat,
                                   hue="algorithm",
                                   capsize=0.1,
                                   err_kws={'linewidth': 1, 'color': 'black'},
                                   errorbar="sd",
                                   palette=color_map)

                plt.ylim(0.0, 1)

                legend = plt.legend(title="\\textbf{Algorithm}", bbox_to_anchor=(1.05, 1), loc='upper left')
                legend.get_frame().set_facecolor('none')
                legend.get_frame().set_edgecolor('none')

                plt.ylabel(stat.capitalize())
                x_label = None
                if x == "variant":
                    x_label = "Experiment Variant"
                plt.xlabel(x_label)
                # plt.title(stat)

                plt.tight_layout()
                plt.savefig(f"{directory}/exp5_3_{stat}_{runtime_experiment}.pdf")
                plt.savefig(f"{figures_directory}/exp5_3_{stat}_{runtime_experiment}.pdf")
                plt.close()

                write_latex_figure_file(figures_directory,
                                        f"exp5_3_{stat}_{runtime_experiment}.pdf",
                                        width=1.0,
                                        caption=caption.replace("_STAT_", stat.capitalize()),
                                        label=f"exp5_3_{stat}_{runtime_experiment}")

if EXPERIMENT_5_4:
    source_directory = "thesis_experiments/experiment_3"
    directory = "thesis_experiments/experiment_5_4"
    if PLOT:
        print("Plotting thesis experiment 5.4")
        sns.set_theme()

        runtime_experiments = [
            ("1_1", "variant", "Runtimes of experiments from \Cref{sec:pretraining-sampled-subgraph}."),
            ("1_2", "variant", "Runtimes of experiments from \Cref{sec:pretraining-smaller-network}."),
            ("3", None, "Runtimes of experiment from \Cref{sec:twitter-dataset-evaluation}."),
            ("4_2", "variant", "Runtimes of experiment from \Cref{sec:pretraining-before-attack}."),
            ("5_1", "graph_size",
             "Runtimes of experiment from \Cref{sec:experiment-scaling-to-very-large-networks}.")
        ]
        for (runtime_experiment, x, caption) in runtime_experiments:
            df_raw = read_experiment_csv(f"thesis_experiments/experiment_{runtime_experiment}", "raw_data.csv")

            if runtime_experiment == "5_1":
                df_raw = df_raw[df_raw["graph_size"].isin([2000, 8000, 32000, 128000])]

            plt.figure(figsize=(10, 4))
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Computer Modern Roman']
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

            plot = sns.barplot(data=df_raw,
                               x=x,
                               y="runtime",
                               hue="algorithm",
                               capsize=0.1,
                               err_kws={'linewidth': 1, 'color': 'black'},
                               errorbar="sd",
                               palette=color_map)

            if runtime_experiment == "5_1":
                plot.set_yscale("log")
            else:
                plt.ylim(0, None)

            legend = plt.legend(title="\\textbf{Algorithm}", bbox_to_anchor=(1.05, 1), loc='upper left')
            legend.get_frame().set_facecolor('none')
            legend.get_frame().set_edgecolor('none')

            plt.ylabel("Runtime (ms)")
            x_label = None
            if x == "variant":
                x_label = "Experiment Variant"
            elif x == "graph_size":
                x_label = "Graph Size"
            plt.xlabel(x_label)
            plt.title("Evaluation Runtime (ms)")

            plt.tight_layout()
            plt.savefig(f"{directory}/exp5_4_runtime_{runtime_experiment}.pdf")
            plt.savefig(f"{figures_directory}/exp5_4_runtime_{runtime_experiment}.pdf")
            plt.close()

            write_latex_figure_file(figures_directory,
                                    f"exp5_4_runtime_{runtime_experiment}.pdf",
                                    width=1.0,
                                    caption=caption,
                                    label=f"exp5_4_runtime_{runtime_experiment}")

print("\nThesis experiments complete.")
