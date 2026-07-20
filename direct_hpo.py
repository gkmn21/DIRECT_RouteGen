'''
This script is used to train a reinforcement learning model on a city environment using start nodes in train set.
'''
#%%
import torch as th
import pandas as pd
import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


import networkx as nx

import numpy as np
import pickle

import os
import gc

from utils import read_data, rescale_inputs, normalise_inputs, distance_matrix_to_poi_graph, delete_study_if_exists
from direct_env import CityEnv

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import argparse

def parse_args_for_hpo_script(args = None):
    '''
    Argument parser
    '''
    parser = argparse.ArgumentParser(
        description = 'Parser for hpo script',
        usage = 'direct_hpo.py [<args>] [-h | --help]'
    )
    parser.add_argument('--city', type = str, default = 'bonn', help = 'Dataset city: berlin, bonn, hamburg')
    parser.add_argument('--poi_graph_threshold', type = int, default = 3000, help = 'POI graph distance threshold in meters')
    parser.add_argument('--cpg_k', type = int, default = 3, help = 'Candidate POI generator k parameter')
    parser.add_argument('--alpha_params', type = str, default = '0.33,0.33,0.33', help = 'Alpha parameters (diversity, coverage, catprefs) for reward function as comma separated values')
    parser.add_argument('--end_node_variant', type = bool, default = False, help = 'Whether to use end node variant')

    return parser.parse_args(args)


#%%
if __name__ == '__main__':
    #%%
    args = parse_args_for_hpo_script()
    print(f'args {args}')
    CITY = args.city
    POI_graph_threshold = args.poi_graph_threshold # in meters
    poi_graph_exists = True # set to false if POI graph needs to be created from distance matrix
    candidate_poi_generator_k = args.cpg_k
    alpha_params = (1,) + tuple([float(x) for x in args.alpha_params.split(',')])
    end_node_variant = args.end_node_variant
    EXP_NAME = f'hpo_exp_{CITY}_dqn_hpo_thres{POI_graph_threshold}_k{candidate_poi_generator_k}_alpha{alpha_params[0]}-{alpha_params[1]}-{alpha_params[2]}-{alpha_params[3]}'
    DATA_DIR = f'./data/{CITY}'
    IMPL_DIR = './content/'
    OUTPUT_DIR = f'./results/{EXP_NAME}'
    VAL_OUTPUT_DIR = f'./results/{EXP_NAME}/val_env_results'
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    os.makedirs(VAL_OUTPUT_DIR, exist_ok = True)
    
    #---------------------------------------------------------------------------------------#
    # read data (start nodes) of training dataset
    TRAIN_SET_PATH = f'./data/{CITY}/saved_data/train_set.csv'
    VAL_SET_PATH = f'./data/{CITY}/saved_data/val_set.csv'
    # dataframe with route requests of test set
    train_set = pd.read_csv(
        TRAIN_SET_PATH
    )
    print(f'test_set.columns {train_set.columns}, test_set.shape {train_set.shape}')
    val_set = pd.read_csv(
        VAL_SET_PATH
    )
    print(f'val_set.columns {val_set.columns}, val_set.shape {val_set.shape}')


    #%%
    # read city data i.e walking network, poi graph, distance and bearing matrix, etc.
    (
        final_pois, _, start_node_pois_within_radius,
        idx2poiid, poiid2idx, unfiltered_distance_matrix, node_category_attr_dict,
        POI_graph, distance_matrix, bearing_matrix
    ) = read_data(DATA_DIR, POI_graph_threshold if poi_graph_exists else None)
    #%%
    #%%
    # if POI graph does not exist, create it from distance matrix
    if not poi_graph_exists:
        POI_graph, distance_matrix, node_category_attr_dict = distance_matrix_to_poi_graph(unfiltered_distance_matrix, final_pois, idx2poiid, threshold = POI_graph_threshold)
        os.makedirs(os.path.join(DATA_DIR, 'saved_data', f'poi_graph_{POI_graph_threshold}'), exist_ok = True)
        
        with open(os.path.join(DATA_DIR, 'saved_data', f'poi_graph_{POI_graph_threshold}', 'node_category_attr_dict.pkl'), 'wb') as f:
            pickle.dump(node_category_attr_dict, f)
        nx.write_graphml(POI_graph, os.path.join(DATA_DIR, 'saved_data', f'poi_graph_{POI_graph_threshold}', 'POI_graph_updated.graphml'))
        with open(os.path.join(DATA_DIR, 'saved_data', f'poi_graph_{POI_graph_threshold}', 'distance_matrix_updated.npy'), 'wb') as f:
            np.save(f, distance_matrix)
    #%%
    (
        POI_graph, final_pois, distance_matrix,
        unfiltered_distance_matrix, start_node_pois_within_radius, _
    ) = rescale_inputs(
        POI_graph,
        final_pois,
        poiid2idx,
        distance_matrix,
        unfiltered_distance_matrix,
        start_node_pois_within_radius
    )

    # now nans are present instead of np.inf
    # distance_matrix, unfiltered_distance_matrix = normalise_inputs(distance_matrix, unfiltered_distance_matrix) # now nans are present instewad of np.inf
    # Note: Only normalising distance_matrix here.
    # 'unfiltered_distance_matrix' is not normalised, it has actual distance values which will be subtracted from dist. budget in the model 
    distance_matrix = normalise_inputs(distance_matrix)
    

    # change poiid2idx after rescaling inputs
    # new poiid2idx where both key and value are ids and not poi ids
    poiid2idx2 = dict.fromkeys(list(poiid2idx.values()))
    for key in poiid2idx2.keys():
        poiid2idx2[key] = key
    print(f'len(poiid2idx2) {len(poiid2idx2)}')

    train_set['start_node_ids'] = train_set['start_node'].apply(lambda x: poiid2idx[x])
    val_set['start_node_ids'] = val_set['start_node'].apply(lambda x: poiid2idx[x])
    if end_node_variant:
        train_set['end_node_ids'] = train_set['end_node'].apply(lambda x: poiid2idx[x])
        val_set['end_node_ids'] = val_set['end_node'].apply(lambda x: poiid2idx[x])

        if CITY == 'verona':
            train_set['cat_prefs'] = None
            val_set['cat_prefs'] = None
    #---------------------------------------------------------------------------------------#

    # register custom environment
    gym.register(
        id = 'gymnasium_env/CityEnv-v0',
        entry_point= CityEnv
    )

    MAX_CITY_GRAPH_NODES = final_pois.shape[0]

    #---------------------------------------------------------------------------------------#
    # hyperparameters config
    N_TRIALS = 100 # Max. number of trials
    N_JOBS = 1 # Number of parallel jobs
    N_STARTUP_TRIALS = 5#10 # Number of initial random trials
    N_EVALUATIONS = 10 # Number of evaluation per trial
    N_TIMESTEPS = int(1e6) # Training budget
    EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
    N_EVAL_ENVS = 5#3 # 5!!
    N_EVAL_EPISODES = len(val_set)
    TIMEOUT = int(60 * 60 * 6)

    def sample_dqn_params(trial):

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log = True)
        buffer_size = trial.suggest_categorical('buffer_size', [50000, 100000, 200000, 500000])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        gamma = trial.suggest_float('gamma', 0.8, 0.9999, log = True)
        train_freq = trial.suggest_categorical('train_freq', [1, 4, 8, 16])
        exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1, step = 0.01)
        exploration_fraction = trial.suggest_float('exploration_fraction', 0.05, 0.75, step = 0.1)

        density_weight = trial.suggest_float('density_weight', 0.0, 1.0, step = 0.25)
        arc_length = trial.suggest_categorical('arc_length', [2500, 3000, 4000, 5000])


        return {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'gamma': gamma,
            'train_freq': train_freq,
            'exploration_final_eps': exploration_final_eps,
            'exploration_fraction': exploration_fraction,
            'density_weight': density_weight,
            'arc_length': arc_length
        }
    
    class TrialEvalCallback(EvalCallback):
        """
        Callback used for evaluating and reporting a trial.
        
        :param eval_env: Evaluation environement
        :param trial: Optuna trial object
        :param n_eval_episodes: Number of evaluation episodes
        :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
        :param deterministic: Whether the evaluation should
            use a stochastic or deterministic policy.
        :param verbose:
        """

        def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
        ):

            super().__init__(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                deterministic=deterministic,
                verbose=verbose,
            )
            self.trial = trial
            self.eval_idx = 0
            self.is_pruned = False

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                # Evaluate policy (done in the parent class)
                super()._on_step()
                self.eval_idx += 1
                # Send report to Optuna
                self.trial.report(self.last_mean_reward, self.eval_idx)
                # Prune trial if need
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False
            return True

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function used by Optuna to evaluate
        one configuration (i.e., one set of hyperparameters).

        Given a trial object, it will sample hyperparameters,
        evaluate it and report the result (mean episodic reward after training)

        :param trial: Optuna trial object
        :return: Mean episodic reward after training
        """


        print(f'Trial {trial.number} starting')
        trial_env_id = f'gymnasium_env/CityEnv{trial.number}-v0'

        # 1. Sample hyperparameters and update the keyword arguments
        sampled_params = sample_dqn_params(trial)
        dqn_params = {
            'learning_rate': sampled_params['learning_rate'],
            'buffer_size': sampled_params['buffer_size'],
            'batch_size': sampled_params['batch_size'],
            'gamma': sampled_params['gamma'],
            'train_freq': sampled_params['train_freq'],
            'exploration_final_eps': sampled_params['exploration_final_eps'],
            'exploration_fraction': sampled_params['exploration_fraction']
        }
        circle_params = {
            'n_cuts': 20,
            'geohash_precision': 6,
            'density_weight': sampled_params['density_weight'],
            'arc_length': sampled_params['arc_length']
        }

        env = gym.make(
            'gymnasium_env/CityEnv-v0',
            city_graph = POI_graph,
            all_start_nodes = start_node_pois_within_radius,
            bearing_matrix = bearing_matrix,
            poiid2idx = poiid2idx2,
            final_pois_gdf = final_pois,
            unfiltered_distance_matrix = unfiltered_distance_matrix,
            output_dir = OUTPUT_DIR,
            train_samples = train_set,
            current_mode = 'train',
            max_city_graph_nodes = MAX_CITY_GRAPH_NODES, #500,
            candidate_poi_generator_k = candidate_poi_generator_k,
            alpha_params_dict = {
                'temporal_distance': alpha_params[0],
                'diversity': alpha_params[1], # 0.33,
                'coverage': alpha_params[2], # 0.33
                'cat_prefs': alpha_params[3] # 0.33
            },
            end_node_variant = end_node_variant,
            circle_params = circle_params
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps = 100)
        
        # Create the RL model
        model = DQN(
            'MultiInputPolicy',
            env,
            verbose = 1,
            **dqn_params
        )

        # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`

        def make_eval_env_factory(train_samples, output_dir):
            # factory creator — returns a function that creates the env when called
            def _init():
                print("[make_eval_env_factory] creating eval env")
                e = CityEnv(
                    city_graph = POI_graph,
                    all_start_nodes = start_node_pois_within_radius,
                    bearing_matrix = bearing_matrix,
                    poiid2idx = poiid2idx2,
                    final_pois_gdf = final_pois,
                    unfiltered_distance_matrix = unfiltered_distance_matrix,
                    output_dir = output_dir,
                    train_samples = train_samples,   # small subset / index list preferred
                    current_mode = 'train',
                    max_city_graph_nodes = MAX_CITY_GRAPH_NODES,
                    candidate_poi_generator_k = candidate_poi_generator_k,
                    alpha_params_dict = {
                        'temporal_distance': alpha_params[0],
                        'diversity': alpha_params[1],
                        'coverage': alpha_params[2],
                        'cat_prefs': alpha_params[3]
                    },
                    end_node_variant = end_node_variant,
                    circle_params = circle_params
                )
                return gym.wrappers.TimeLimit(e, max_episode_steps=100)
            return _init

        # Create N_EVAL_ENVS factories and build a single-process vector env
        eval_factories = [make_eval_env_factory(val_set, VAL_OUTPUT_DIR) for _ in range(N_EVAL_ENVS)]
        eval_envs = DummyVecEnv(eval_factories)
        print("[objective] eval_envs created (DummyVecEnv)")

        eval_callback = TrialEvalCallback(
            eval_envs,
            trial,
            N_EVAL_EPISODES,
            EVAL_FREQ,
            True,
            1
        )


        nan_encountered = False
        try:
            # Train the model
            model.learn(N_TIMESTEPS, callback=eval_callback)
        # except AssertionError as e:
        except Exception as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True
        finally:
            # Free memory

            model.env.close()
            eval_envs.close()
            env.close()
            del model, eval_envs, env
            gc.collect()

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward
        

    # Set pytorch num threads to 1 for faster training
    th.set_num_threads(1)
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    # Delete previous study if it exists
    delete_study_if_exists(EXP_NAME, "sqlite:///db.sqlite3")
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(
        study_name=EXP_NAME,
        storage="sqlite:///db.sqlite3",
        sampler=sampler,
        pruner=pruner,
        direction="maximize"
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv(f"study_results_{EXP_NAME}.csv")



    #########################################################################################
    