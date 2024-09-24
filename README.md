# Sybil Detection using Graph Neural Networks

This repository contains the code for the Master Thesis "Sybil Detection using Graph Neural Networks" by Stuart Heeb, done under the supervision of Andreas Plesner and Prof. Dr. Wattenhofer at the [Distributed Computing Group](https://disco.ethz.ch) of ETH Zürich.

This thesis was also submitted as a paper to the AAAI 2025 conference.

## Quick Links

- [Paper (pdf)](paper) (to be added)
- [Thesis (pdf)](thesis/Stuart_Heeb_Sybil_Detection_using_GNNs.pdf)
- [Presentation (pdf)](presentation/presentation.pdf)

## Installation Guide

1. Install the conda environment with the provided `.yml` file, while in the root directory of the repository:

```
conda env create -f environment.yml
```

2. Activate the environment:

```
conda activate gnn-sybil-detection
```


3. Change into the `code` directory

```
cd code
```

4. Run the experiments

```
python paper_experiments.py  # For the paper experiments
python thesis_experiments.py  # For the thesis experiments
python thesis_figures.py  # For the remaining thesis figures
```



## Complete Thesis Experiment Data

### Experiment 1

- Experiment 1.1: [raw_data.csv](code/thesis_experiments/experiment_1_1/raw_data.csv)
- Experiment 1.2: [raw_data.csv](code/thesis_experiments/experiment_1_2/raw_data.csv)

### Experiment 2

Evaluation of

- Data model: [raw_data.csv](code/thesis_experiments/experiment_2/data_model_column_raw_data.csv)
- Network size: [raw_data.csv](code/thesis_experiments/experiment_2/graph_size_column_raw_data.csv)
- Train nodes fraction:  [raw_data.csv](code/thesis_experiments/experiment_2/train_nodes_fraction_column_raw_data.csv)
- Train label noise fraction: [raw_data.csv](code/thesis_experiments/experiment_2/label_noise_fraction_column_raw_data.csv)

### Experiment 3

[raw_data.csv](code/thesis_experiments/experiment_3/raw_data.csv)

### Experiment 4

- Experiment 4.1: [raw_data.csv](code/thesis_experiments/experiment_4_1/raw_data.csv)
- Experiment 4.2: [raw_data.csv](code/thesis_experiments/experiment_4_2/raw_data.csv)

### Experiment 5

- Experiment 5.1: [raw_data.csv](code/thesis_experiments/experiment_5_1/raw_data.csv)
- Experiment 5.2: [raw_data.csv](code/thesis_experiments/experiment_5_2/raw_data.csv)
