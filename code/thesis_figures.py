import numpy as np
import seaborn as sns
from social_networks import *
from graphs import *
from latex_utils import *

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Power law distribution
if True:
    a = 68.0
    c = 1.0
    k = 1.5
    x = np.linspace(1, 50, 1000, dtype=float)
    y = a * (c * x) ** (-k)
    df_PL_dist = pd.DataFrame({'x': x, 'y': y})
    plt.figure(figsize=(5, 5))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    sns.lineplot(
        df_PL_dist, x="x", y="y"
    )
    plt.xlim(0, 50)
    plt.savefig(f"thesis_experiments/thesis_figures/power-law-distribution.pdf")
    write_latex_figure_file(directory="thesis_experiments/thesis_figures",
                            figure_file_name="power-law-distribution.pdf",
                            width=0.4,
                            caption="Power Law distribution.",
                            label="power-law-distribution")
    plt.close()

# Power law graph
if True:
    m = 10
    p = 0.8
    graph = PowerLawGraph(4039, m, p)
    degrees = dict(graph.graph.degree())
    df_PL = pd.DataFrame(list(degrees.items()), columns=['node', 'degree'])
    # plt.figure(figsize=(4, 2))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    sns.displot(
        df_PL, x="degree", binwidth=5, height=2.5, aspect=2
    )
    plt.xlim(0, 250)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.savefig(f"thesis_experiments/thesis_figures/power-law-graph.pdf")
    write_latex_figure_file(directory="thesis_experiments/thesis_figures",
                            figure_file_name="power-law-graph.pdf",
                            width=0.65,
                            caption="Node degree distribution of a Power Law graph with 4039 nodes, $m=10$ and $p=0.8$.",
                            label="power-law-graph")
    plt.close()

# Barabasi Albert graph
if True:
    m = 10
    graph = BarabasiAlbertGraph(4039, m)
    degrees = dict(graph.graph.degree())
    df_BA = pd.DataFrame(list(degrees.items()), columns=['node', 'degree'])
    # plt.figure(figsize=(4, 2))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    sns.displot(
        df_BA, x="degree", binwidth=5, height=2.5, aspect=2
    )
    plt.xlim(0, 250)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.savefig(f"thesis_experiments/thesis_figures/barabasi-albert-graph.pdf")
    write_latex_figure_file(directory="thesis_experiments/thesis_figures",
                            figure_file_name="barabasi-albert-graph.pdf",
                            width=0.65,
                            caption="Node degree distribution of a Barabási-Albert graph with 4039 nodes and $m=10$.",
                            label="barabasi-albert-graph")
    plt.close()

# Facebook graph node degree distribution
if True:
    FB = FacebookSNAP()
    degrees = dict(FB.graph.degree())
    df_FB = pd.DataFrame(list(degrees.items()), columns=['node', 'degree'])
    # plt.figure(figsize=(4, 2))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    sns.displot(
        df_FB, x="degree", binwidth=10, height=2.5, aspect=2
    )
    plt.xlim(0, 500)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.savefig(f"thesis_experiments/thesis_figures/facebook-degree-distribution.pdf")
    write_latex_figure_file(directory="thesis_experiments/thesis_figures",
                            figure_file_name="facebook-degree-distribution.pdf",
                            width=0.65,
                            caption="Node degree distribution of the Facebook graph.",
                            label="facebook-degree-distribution")
    plt.close()

# FB social network node degree distribution
if True:
    FB = SocialNetworkFromRegions(honest_region=FacebookSNAP(), sybil_region=FacebookSNAP(is_sybil=True))
    print(FB.network.graph)
    RandomAttack().perform_attack(social_network=FB, num_attack_edges=4 * 4039)
    print(FB.network.graph)

    degrees = dict(FB.network.graph.degree())
    df_FB = pd.DataFrame(list(degrees.items()), columns=['node', 'degree'])
    # plt.figure(figsize=(4, 2))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    sns.displot(
        df_FB, x="degree", binwidth=10, height=2.5, aspect=2
    )
    plt.xlim(0, 500)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.savefig(f"thesis_experiments/thesis_figures/FB-social-network-degree-distribution.pdf")
    write_latex_figure_file(directory="thesis_experiments/thesis_figures",
                            figure_file_name="FB-social-network-degree-distribution.pdf",
                            width=0.65,
                            caption="Node degree distribution of a social network synthesized with the Facebook graph (\Cref{sec:facebook-graph}) by adding 4 random attack edges per Sybil.",
                            label="FB-social-network-degree-distribution")
    plt.close()

# PL social network node degree distribution
if True:
    PL_SN = SocialNetworkFromRegions(honest_region=PowerLawGraph(4039, 10, 0.8),
                                     sybil_region=PowerLawGraph(4039, 10, 0.8, is_sybil=True))
    RandomAttack().perform_attack(social_network=PL_SN, num_attack_edges=4 * 4039)

    degrees = dict(PL_SN.network.graph.degree())
    df_PL_SN = pd.DataFrame(list(degrees.items()), columns=['node', 'degree'])
    # plt.figure(figsize=(4, 2))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    sns.displot(
        df_PL_SN, x="degree", binwidth=5, height=2.5, aspect=2
    )
    plt.xlim(0, 250)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.savefig(f"thesis_experiments/thesis_figures/PL-social-network-degree-distribution.pdf")
    write_latex_figure_file(directory="thesis_experiments/thesis_figures",
                            figure_file_name="PL-social-network-degree-distribution.pdf",
                            width=0.65,
                            caption="Node degree distribution of a social network synthesized with a graph generated by the Power Law model (\Cref{sec:power-law-model}) with parameters $m=10$ and $p=0.8$, by adding 4 random attack edges per Sybil.",
                            label="PL-social-network-degree-distribution")
    plt.close()
