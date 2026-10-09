# DIRECT: Deep Reinforcement Learning for Tourist Route Generation


The code in this repository is an implementation of our published approach: <br>
Genivika Mann, Rajjat Dadwal, and Elena Demidova. “DIRECT: Deep Reinforcement Learning for Tourist Route Generation”. In: ACM Transactions on Spatial Algorithms and Systems (TSAS), 2026. Association for Computing Machinery. DOI: https://doi.org/10.1145/3839237. <br>
If you use the source code, please cite our research (see [Citation](#citation)). <br>

## Notes

1) This repository does not contain any data. For Berlin, Bonn, Hamburg, New York and Tokyo, all POI and street network data is fetched from [OpenStreetMap](https://www.openstreetmap.org/) by the provided data generation script. For Verona, the Verona dataset from Vecchia et al. has to be obtained separately and its location set in `DATASET_PATH` in `dataset_generation/prepare_verona_data.py`. <br>
2) The data generation scripts write to `../data/<city>/` relative to the current working directory, so they must be run from inside the `dataset_generation/` folder. The training scripts read from `./data/<city>/` and must be run from the repository root. <br>

## Introduction
The provided source code generates walking tourist routes using deep reinforcement learning. Given a user request consisting of a start POI, an end POI, a time budget and optional category preferences, DIRECT generates a walkable route through points of interest (POIs) of a city that respects the time budget while balancing route diversity, POI coverage and the user's category preferences.

## Implementation Overview
The source code consisting of dataset generation, (optional) hyperparameter optimization, model training and evaluation. In the dataset generation stage, POIs and the walkable street network of a city are fetched from OpenStreetMap, a POI graph with walking distances is built and training, validation and test user requests are created. Route generation is modelled as a [Gymnasium](https://gymnasium.farama.org/) environment (`direct_env.py`) in which a candidate POI generator proposes the next POIs to visit and a reinforcement learning agent from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) selects among the POIs. The trained model is evaluated on the test set and the evaluation metrics are computed overall and per time budget group. <br>

## Project Directory Structure
When the listed commands are run sequentially, all required folders are created in the structure shown below. Generated data, model checkpoints, logs and evaluation results are stored in the `data/`, `content/` and `results/` folders in the repository root. <br>

```
DIRECT_RouteGen/
│
├── dataset_generation/
│   ├── prepare_data.py
│   ├── prepare_verona_data.py
│   └── constants.py
│
├── constants.py
├── direct_env.py
├── direct_hpo.py
├── train_eval_model.py
├── eval.py
├── utils.py
│
├── data/
│   └── <city>/
│       └── saved_data/
│           ├── final_pois.csv
│           ├── walking_network.graphml
│           ├── distance_matrix.npy
│           ├── distance_matrix_updated.npy
│           ├── bearing_matrix.npy
│           ├── POI_graph_updated.graphml
│           ├── indexing_dicts.pkl
│           ├── node_category_attr_dict.pkl
│           ├── path_related_dicts.pkl
│           ├── start_node_pois_within_radius.pkl
│           ├── train_start_node_pois.pkl
│           ├── test_start_node_pois.pkl
│           ├── train_set.csv
│           ├── val_set.csv
│           ├── test_set.csv
│           └── poi_graph_<threshold>/
│
├── content/
│   ├── models/
│   │   └── <exp_name>/
│   │       └── <timestep>.zip
│   └── logs/
│
└── results/
    └── <exp_name>/
        ├── episode_results.csv
        ├── {global|2h_to_4h|5h_to_7h|8h_to_10h}_intermediate_results.csv
        ├── <exp_name>_rep_metrics.csv
        └── test_env_results/

```

## Folder Descriptions
<b> dataset_generation/ </b> <br>
Contains the scripts for creating the city datasets. <br>
  + prepare_data.py – fetches POIs and the walkable street network from OpenStreetMap and creates the user requests for Berlin, Bonn, Hamburg, New York and Tokyo
  + prepare_verona_data.py – creates the Verona dataset from the Verona dataset of Vecchia et al., with the street network from OpenStreetMap
  + constants.py – OpenStreetMap tags and POI categories used for data extraction

<b> data/&lt;city&gt;/saved_data/ </b> <br>
Stores the generated dataset of a city. Important files include: <br>
  + final_pois.csv – the POIs of the city with their categories and attributes
  + walking_network.graphml – the walkable street network
  + distance_matrix.npy, distance_matrix_updated.npy – walking distances between POIs
  + bearing_matrix.npy – bearings between POIs
  + POI_graph_updated.graphml – POI graph connecting POIs within a distance threshold
  + indexing_dicts.pkl, node_category_attr_dict.pkl, path_related_dicts.pkl – lookup dictionaries for POI ids, categories and paths
  + train_set.csv, val_set.csv, test_set.csv – training, validation and test user requests
  + poi_graph_&lt;threshold&gt;/ – POI graph and distance matrix for a given POI graph distance threshold

<b> content/ </b> <br>
Contains the DIRECT model checkpoints and training logs:
  + models/&lt;exp_name&gt;/ – model checkpoints saved during training
  + logs/ – training logs

<b> results/&lt;exp_name&gt;/ </b> <br>
Stores the test set evaluation results, including:
  + episode_results.csv – the generated route for each test request
  + {variant}_intermediate_results.csv – per-request evaluation details for all test requests (global) and for small (2–4 h), medium (5–7 h) and large (8–10 h) time budgets
  + &lt;exp_name&gt;_rep_metrics.csv – the aggregated evaluation metrics per variant

The experiment name &lt;exp_name&gt; encodes the city, candidate POI generator k, alpha parameters and further settings of the run. <br>


<b> Repository Usage: </b> <br>
All python packages can be installed in a virtual environment by running the following command in the terminal:
```
pip install -r requirements.txt
```
Main dependencies include Python 3.12.4, PyTorch 2.7.0, `stable-baselines3` and `gymnasium`.

<b> Dataset Generation: </b> <br>
The following python command, run from inside the `dataset_generation/` folder, fetches the POIs and street network of a city from OpenStreetMap and creates the training, validation and test set user requests: <br>
```
python3 prepare_data.py --city '<city_name>'
```
where &lt;city_name&gt; is one of berlin, bonn, hamburg, new york or tokyo. The dataset is saved in `data/<city_name>/saved_data/`. <br>
For the Verona dataset, set `DATASET_PATH` in `prepare_verona_data.py` to the location of the Verona dataset and run:
```
python3 prepare_verona_data.py
```

<b> Hyperparameter Optimization (optional): </b> <br>
Hyperparameters of the DQN agent can be tuned with Optuna by running the following command from the repository root: 
```
python3 direct_hpo.py --city '<city_name>;' --cpg_k '<cpg_k>' --alpha_params '0.33,0.33,0.33'
```
The POI graph distance threshold (in meters) can be set with `--poi_graph_threshold` (default 3000). The Optuna study is stored in `db.sqlite3` and the trial results are saved to `study_results_<exp_name>.csv`. <br>

<b> DIRECT Model Training and Evaluation: </b> <br>
Once the dataset of a city is created, the model can be trained on the training set and evaluated on the test set by running the following command from the repository root: <br>
```
python3 train_eval_model.py --city '<city_name>' --cpg_k '<cpg_k>' --alpha_params '0.33,0.33,0.33'
```
where &lt;city_name&gt; is one of berlin, bonn, hamburg, new york, tokyo or verona, &lt;cpg_k&gt; is the candidate POI generator k parameter (default 3) and the alpha parameters (diversity, coverage, category preferences) weight the reward function, given as comma separated values. <br>
The model checkpoints and logs are saved in the `content/` folder. The test set evaluation metrics and generated routes are saved in the `results/` folder. <br>

## Disclaimer
+ The OpenStreetMap name is a trademark of the OpenStreetMap Foundation and is used with their permission. We are not endorsed by or affiliated with the OpenStreetMap Foundation.
+ This code  is published only to support reproducibility in research. It is not intended for any type of commercial use or placement on the market.

## Licenses
  + The files `dataset_generation/constants.py` and `constants.py` contain OpenStreetMap tags. OpenStreetMap data is © OpenStreetMap contributors, made available under the terms of the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/)https://opendatacommons.org/licenses/odbl/ by the OpenStreetMap Foundation (OSMF) https://www.openstreetmap.org/copyright.
  + The code in this repository, except for the OpenStreetMap material identified above, is licensed under the MIT license.

## Affiliations
The source code is being published in affiliation with the University of Bonn, Germany (https://www.uni-bonn.de/de) and the Lamarr Institute for Machine Learning and Artificial Intelligence, Bonn, Germany (https://lamarr-institute.org/).

## Acknowledgements
This work was partially funded by mFUND, Federal Ministry of Transport (BMV), Germany ('MoToRes', 01F2271C), the Deutsche Forschungsgemeinschaft (DFG) ('RUSHMORE', 547610252) and the Lamarr Institute for Machine Learning and Artificial Intelligence, Bonn, Germany (https://lamarr-institute.org/).

## Citation
If you find this work useful in your research, please cite:

```bibtex
@article{10.1145/3839237,
author = {Mann, Genivika and Dadwal, Rajjat and Demidova, Elena},
title = {DIRECT: Deep Reinforcement Learning for Tourist Route Generation},
year = {2026},
publisher = {Association for Computing Machinery},
issn = {2374-0353},
doi = {10.1145/3839237},
note = {Just Accepted},
journal = {ACM Transactions on Spatial Algorithms and Systems}
}
```


