# DMT for Panacea

This is the official repository for NOCTOWL :owl: (Network intrusiOn deteCTiOn With online tree-based Learning).
NOCTOWL is a lightweight, explainable anomaly detection model based on binary decision trees.
It enhances the base tree with leaf-node clustering, batch-wise analysis, and sampling strategies to detect anomalies and concept drift.
This repository contains the **code** and **instructions** to reproduce the experiments.

## Datasets
The datasets are available at this [link](https://drive.google.com/drive/folders/1PG_tPCxmS2rdkIMokjBnQkXhIJgJJlEY). 

Put all the downloaded csv files in the `./datesets/ ` folder.

## Requirements
Before running the experiments, please install the requirements:

```bash
pip install -r requirements.txt
```

All experiments have been tested with Python 3.11.2.

## Run the experiments

To run the experiments, execute the following command:

```bash
python3 main.py --dataset <name> [OPTIONS]
```

The available options are:
- `--dataset`: to specify the name of the dataset
- `--mode`: to specify the run name
- `--max_depth`: max depth of the tree
- `--dist_threshold`: threshold to determine whether the batch data are useful or should be discarded.
- `--homogeneity_gain_threshold`: minimum homogeneity gain required to split a node.
- `--min_point_per_leaf`: minimum number of samples required per leaf.
- `--delay`: number of batches to wait before obtaining the labels.
- `--sampling`: fraction of samples to use for tree update during training.
