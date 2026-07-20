'''
This script is used to train a reinforcement learning model on a city environment
with best trial hyperparameters and evaluate the test set.
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

import osmnx as ox
import networkx as nx
from shapely.geometry import Point, MultiPolygon, LineString, Polygon
from shapely.ops import split
import numpy as np
import pickle

import matplotlib.pyplot as plt
import geopandas as gpd
import pygeohash as pgh

from enum import Enum
from haversine import haversine, Unit

from tqdm import tqdm

import folium
from shapely import wkt
from math import isnan
import time
import os
import ast
import gc

from collections import Counter
from utils import read_data, rescale_inputs, normalise_inputs, distance_matrix_to_poi_graph, delete_study_if_exists
from constants import FINAL_CATEGORIES
from dataset_generation.constants import NEW_VISIT_DURATION_BASED_ON_CATEGORIES
from direct_env import CityEnv
from eval import compute_metrics_simplified_with_end_node
import argparse

def parse_args_for_train_eval_script(args = None):
    '''
    Argument parser
    '''
    parser = argparse.ArgumentParser(
        description = 'Parser for training and eval script',
        usage = 'train_eval_model.py [<args>] [-h | --help]'
    )
    parser.add_argument('--city', type = str, default = 'bonn', help = 'Dataset city: berlin, bonn, hamburg, new york, tokyo, verona')
    parser.add_argument('--cpg_k', type = int, default = 3, help = 'Candidate POI generator k parameter')
    parser.add_argument('--alpha_params', type = str, default = '0.33,0.33,0.33', help = 'Alpha parameters (diversity, coverage, catprefs) for reward function as comma separated values')

    return parser.parse_args(args)

# Dictionary of best hyperparameters obtained from hyperparameter optimization
# using Optuna for DQN model on different cities and alpha parameters
best_params_dict = {
    'berlin': {
        'learning_rate': 0.000410215811728085,
        'buffer_size': 500000,
        'batch_size': 128,
        'gamma': 0.9111581459013154,
        'train_freq': 1,
        'exploration_final_eps': 0.04,
        'exploration_fraction': 0.25,
        'poi_graph_threshold': 3000,
        'end_node_variant': False,
        'density_weight': 0.75,
        'arc_length': 5000
    },
    'bonn': {
        'learning_rate': 0.0006145230419253543,
        'buffer_size': 500000,
        'batch_size': 128,
        'gamma': 0.8315144813165809,
        'train_freq': 1,
        'exploration_final_eps': 0.06999999999999999,
        'exploration_fraction': 0.6500000000000001,
        'poi_graph_threshold': 3000,
        'end_node_variant': False,
        'density_weight': 0.75,
        'arc_length': 4000
    },
    'hamburg': {
        'learning_rate': 0.0005792279068399921,
        'buffer_size': 100000,
        'batch_size': 128,
        'gamma': 0.8162390907897724,
        'train_freq': 4,
        'exploration_final_eps': 0.01,
        'exploration_fraction': 0.15000000000000002,
        'poi_graph_threshold': 3000,
        'end_node_variant': False,
        'density_weight': 0,
        'arc_length': 4000
    },
    'new york': {
        'learning_rate': 2.1755859714545845e-05,
        'buffer_size': 500000,
        'batch_size': 128,
        'gamma': 0.9605391108958484,
        'train_freq': 1,
        'exploration_final_eps': 0.06999999999999999,
        'exploration_fraction': 0.55,
        'poi_graph_threshold': 1000,
        'end_node_variant': False,
        'density_weight': 0.25,
        'arc_length': 2500
    },
    'tokyo': {
        'learning_rate':  0.00010401978049739222,
        'buffer_size': 200000,
        'batch_size': 128,
        'gamma': 0.8112522042308962,
        'train_freq': 4,
        'exploration_final_eps': 0.05,
        'exploration_fraction': 0.45,
        'poi_graph_threshold': 1000,
        'end_node_variant': False,
        'density_weight': 0.25,
        'arc_length': 2500
    },
    'verona': {
        'learning_rate': 0.0007805572705402524,
        'buffer_size': 500000,
        'batch_size': 64,
        'gamma': 0.9653977377613575,
        'train_freq': 1,
        'exploration_final_eps': 0.05,
        'exploration_fraction': 0.75,
        'poi_graph_threshold': 5000,
        'end_node_variant': True,
        'density_weight': 0.5,
        'arc_length': 3000
    }
}


#%%
if __name__ == '__main__':
    #%%
    SEED = 100
    np.random.seed(SEED)

    args = parse_args_for_train_eval_script()
    print(f'args {args}')
    CITY = args.city
    end_node_variant =  best_params_dict[f'{CITY}']['end_node_variant']
    POI_graph_threshold =  best_params_dict[f'{CITY}']['poi_graph_threshold'] # in meters
    poi_graph_exists = True
    candidate_poi_generator_k = args.cpg_k #3
    alpha_params = (1,) + tuple([float(x) for x in args.alpha_params.split(',')])
   
    # tuned params
    learning_rate = best_params_dict[f'{CITY}']['learning_rate']
    buffer_size = best_params_dict[f'{CITY}']['buffer_size']
    batch_size = best_params_dict[f'{CITY}']['batch_size']
    gamma = best_params_dict[f'{CITY}']['gamma']
    train_freq = best_params_dict[f'{CITY}']['train_freq']
    exploration_final_eps = best_params_dict[f'{CITY}']['exploration_final_eps']
    exploration_fraction = best_params_dict[f'{CITY}']['exploration_fraction']

    # experiment name and directories
    circle_params = {
        'density_weight': best_params_dict[f'{CITY}']['density_weight'],
        'n_cuts': 20,
        'arc_length': best_params_dict[f'{CITY}']['arc_length'],
        'geohash_precision': 6
    }
    EXP_NAME = f'exp_dw{circle_params["density_weight"]}_ncuts{circle_params["n_cuts"]}_hashprec{circle_params["geohash_precision"]}_arc{circle_params["arc_length"]}_{CITY}_dqn_thres{POI_graph_threshold}_k{candidate_poi_generator_k}_alpha{alpha_params[0]}-{alpha_params[1]}-{alpha_params[2]}-{alpha_params[3]}_{SEED}'
    DATA_DIR = f'./data/{CITY}'
    IMPL_DIR = './content/'
    OUTPUT_DIR = f'./results/{EXP_NAME}'
    TEST_OUTPUT_DIR = f'./results/{EXP_NAME}/test_env_results'
    MODEL_ITERATION = '990000.zip'
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok = True)
    
    #---------------------------------------------------------------------------------------#
    # read data (start nodes) of training dataset
    TRAIN_SET_PATH = f'./data/{CITY}/saved_data/train_set.csv'
    TEST_SET_PATH = f'./data/{CITY}/saved_data/test_set.csv'
    # dataframe with route requests of train, test set
    train_set = pd.read_csv(
        TRAIN_SET_PATH
    )
    print(f'train_set.columns {train_set.columns}, train_set.shape {train_set.shape}')
    test_set = pd.read_csv(
        TEST_SET_PATH
    )
    print(f'test_set.columns {test_set.columns}, test_set.shape {test_set.shape}')

    # dataframe to store model results on test set
    results_df = pd.DataFrame(
        columns = ['req_id', 'start_node', 'end_node', 'time_constraint', 'route', 'cum_reward', 'rewards_list', 'action_return_tuple', 'inference_time']
    )


    #%%
    # read city data i.e walking network, poi graph, distance and bearing matrix, etc.
    (
        final_pois, _, start_node_pois_within_radius,
        idx2poiid, poiid2idx, unfiltered_distance_matrix, node_category_attr_dict,
        POI_graph, distance_matrix, bearing_matrix
    ) = read_data(DATA_DIR, POI_graph_threshold if poi_graph_exists else None)
    
    #%%
    # If POI graph does not exist for the given threshold, create and save it
    if not poi_graph_exists:
        POI_graph, distance_matrix, node_category_attr_dict = distance_matrix_to_poi_graph(unfiltered_distance_matrix, final_pois, idx2poiid, threshold = POI_graph_threshold)
        os.makedirs(os.path.join(DATA_DIR, 'saved_data', f'poi_graph_{POI_graph_threshold}'), exist_ok = True)
        
        with open(os.path.join(DATA_DIR, 'saved_data', f'poi_graph_{POI_graph_threshold}', 'node_category_attr_dict.pkl'), 'wb') as f:
            pickle.dump(node_category_attr_dict, f)
        nx.write_graphml(POI_graph, os.path.join(DATA_DIR, 'saved_data', f'poi_graph_{POI_graph_threshold}', 'POI_graph_updated.graphml'))
        with open(os.path.join(DATA_DIR, 'saved_data', f'poi_graph_{POI_graph_threshold}', 'distance_matrix_updated.npy'), 'wb') as f:
            np.save(f, distance_matrix)
        print(f'Saved POI graph for threshold {POI_graph_threshold} meters')
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
    # Note: distance matrices are rescaled to km here after rescale_inputs function

    
    # Note: Only normalising distance_matrix here.
    # 'unfiltered_distance_matrix' is not normalised, it has actual distance values which will be subtracted from dist. budget in the model 
    distance_matrix = normalise_inputs(distance_matrix)
    # now nans are present instead of np.inf in distance matrix
    

    # change poiid2idx after rescaling inputs
    # new poiid2idx where both key and value are ids and not poi ids
    poiid2idx2 = dict.fromkeys(list(poiid2idx.values()))
    for key in poiid2idx2.keys():
        poiid2idx2[key] = key
    print(f'len(poiid2idx2) {len(poiid2idx2)}')

    train_set['start_node_ids'] = train_set['start_node'].apply(lambda x: poiid2idx[x])
    test_set['start_node_ids'] = test_set['start_node'].apply(lambda x: poiid2idx[x])
    if end_node_variant:
        train_set['end_node_ids'] = train_set['end_node'].apply(lambda x: poiid2idx[x])
        test_set['end_node_ids'] = test_set['end_node'].apply(lambda x: poiid2idx[x])

        if CITY == 'verona':
            train_set['cat_prefs'] = None
            test_set['cat_prefs'] = None
    #%%
    #---------------------------------------------------------------------------------------#

    # register custom environment
    gym.register(
        id = 'gymnasium_env/CityEnv-v0',
        entry_point= CityEnv
    )
    #%%

    # make gym environment
    MAX_CITY_GRAPH_NODES = final_pois.shape[0]
    train_env = gym.make(
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
        max_city_graph_nodes = MAX_CITY_GRAPH_NODES,
        candidate_poi_generator_k = candidate_poi_generator_k,
        alpha_params_dict = {
            'temporal_distance': alpha_params[0],
            'diversity': alpha_params[1], # 0.33,
            'coverage': alpha_params[2], # 0.33
            'cat_prefs': alpha_params[3] # 0.33
        },
        end_node_variant = end_node_variant,
        circle_params = circle_params,
        render_neighbors = False if CITY in ['tokyo', 'new york'] else True
    )
    train_env = gym.wrappers.TimeLimit(train_env, max_episode_steps = 100)

    print('checking env with gym and stable baselines............')
    gym.utils.env_checker.check_env(train_env.unwrapped)
    check_env(train_env)

    ##
    # Testing environment with random actions
    ##
    obs = train_env.reset()[0]

    # Take some random actions
    for i in range(10):

        rand_action = train_env.action_space.sample()
        obs, reward, terminated, truncated, _ = train_env.step(rand_action)

        if terminated or truncated:
            # Render Change
            if (i%2 == 0):
                train_env.render()
            train_env.reset()

    #---------------------------------------------------------------------------------------#
    # Train model
    print('Train model')
    model_name = EXP_NAME
    LOG_TIMESTEPS = int(1e4)
    TIMESTEPS = int(1e6)
    models_dir = os.path.join(IMPL_DIR, 'models', model_name)
    log_dir = os.path.join(IMPL_DIR, 'logs')
    print(f'{model_name}, {models_dir}, {log_dir}')
    print(f'TIMESTEPS {TIMESTEPS}, LOG_TIMESTEPS {LOG_TIMESTEPS}')

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_env.reset()
    model = DQN(
        'MultiInputPolicy',
        train_env,
        verbose = 1,
        learning_rate = learning_rate,
        buffer_size = buffer_size,
        batch_size = batch_size,
        gamma = gamma,
        train_freq = train_freq,
        exploration_final_eps = exploration_final_eps,
        exploration_fraction = exploration_fraction,
        tensorboard_log = log_dir
    )

    # Record training and inference time
    start_event = th.cuda.Event(enable_timing = True)
    end_event = th.cuda.Event(enable_timing = True)
    start_event.record()
    for i in tqdm(range(1, int(TIMESTEPS/LOG_TIMESTEPS) + 1)):
        model.learn(
            total_timesteps = LOG_TIMESTEPS,
            reset_num_timesteps = False,
            tb_log_name = f"{model_name}"
        )
        model.save(f"{models_dir}/{LOG_TIMESTEPS*i}")
    end_event.record()
    th.cuda.synchronize()
    training_time = start_event.elapsed_time(end_event)/1000 # in seconds
    print('Training complete.......')
    print(f'Training time: {training_time} seconds')
    #%%
    # ---------------------------------------------------------------------------------------#
    # Evaluate on test set
    print('Evaluate on test set')

    # make gym environment
    test_env = gym.make(
        'gymnasium_env/CityEnv-v0',
        city_graph = POI_graph,
        all_start_nodes = start_node_pois_within_radius,
        bearing_matrix = bearing_matrix,
        poiid2idx = poiid2idx2,
        final_pois_gdf = final_pois,
        unfiltered_distance_matrix = unfiltered_distance_matrix,
        output_dir = OUTPUT_DIR,
        current_mode = 'test',
        max_city_graph_nodes = MAX_CITY_GRAPH_NODES,
        candidate_poi_generator_k = candidate_poi_generator_k,
        alpha_params_dict = {
            'temporal_distance': alpha_params[0],
            'diversity': alpha_params[1], # 0.33,
            'coverage': alpha_params[2], # 0.33
            'cat_prefs': alpha_params[3] #0.33
        },
        end_node_variant = end_node_variant,
        circle_params = circle_params,
        render_neighbors = False if CITY in ['tokyo', 'new york'] else True
    )
    test_env = gym.wrappers.TimeLimit(test_env, max_episode_steps = 100)

    print('checking env with gym and stable baselines............')
    gym.utils.env_checker.check_env(test_env.unwrapped)
    check_env(test_env)

    ##
    # Testing environment with random actions
    ##
    if end_node_variant:
        obs = test_env.reset(
            options = {'test_sample_parameters': {
                'start_node': poiid2idx[test_set.iloc[0]['start_node']],
                'end_node': poiid2idx[test_set.iloc[0]['end_node']],
                'time_constraint': 3,
                'request_id': '',
                'cat_prefs': None
            }}
        )[0]
    else:
        obs = test_env.reset(
            options = {'test_sample_parameters': {
                'start_node': poiid2idx[test_set.iloc[0]['start_node']],
                'time_constraint': 3,
                'request_id': '',
                'cat_prefs': None
            }}
        )[0]
    #%%
    # Take some random actions
    for i in range(10):

        rand_action = test_env.action_space.sample()
        obs, reward, terminated, truncated, _ = test_env.step(rand_action)
        print(f'rand_action {rand_action}, reward {reward}, terminated {terminated}, truncated {truncated}')
        if terminated or truncated:
            # Render Change
            if (i%2 == 0):
                test_env.render()
            
            if end_node_variant:
                test_env.reset(
                    options = {'test_sample_parameters': {
                        'start_node': poiid2idx[test_set.iloc[0]['start_node']],
                        'end_node': poiid2idx[test_set.iloc[0]['end_node']],
                        'time_constraint': 3,
                        'request_id': '',
                        'cat_prefs': None
                    }}
                )
            else:
                test_env.reset(
                    options = {'test_sample_parameters': {
                        'start_node': poiid2idx[test_set.iloc[0]['start_node']],
                        'time_constraint': 3,
                        'request_id': '',
                        'cat_prefs': None
                    }}
                )


    #---------------------------------------------------------------------------------------#
    #%%
    # Run inference
    print('Run inference')
    model_name = EXP_NAME
    models_dir = os.path.join(IMPL_DIR, 'models', model_name)
    model_path = os.path.join(models_dir, MODEL_ITERATION)

    # load model
    model = DQN.load(model_path, env = test_env)

    for idx, ep in tqdm(test_set.iterrows(), total = len(test_set)):

        if end_node_variant:
            obs, info = test_env.reset(
                options = {
                    'test_sample_parameters': {
                        'start_node': poiid2idx[ep['start_node']],
                        'end_node': poiid2idx[ep['end_node']],
                        'time_constraint': ep['time_constraint'],
                        'cat_prefs': ep['cat_prefs'],
                        'request_id': ep['req_id']
                    }
                }
            )
        else:
            obs, info = test_env.reset(
                options = {
                    'test_sample_parameters': {
                        'start_node': poiid2idx[ep['start_node']],
                        'time_constraint': ep['time_constraint'],
                        'cat_prefs': ep['cat_prefs'],
                        'request_id': ep['req_id']
                    }
                }
            )
        done = False
        truncated = False
        cum_reward = None
        rewards_list = []
        action_return_tuples = []
        start_event = th.cuda.Event(enable_timing = True)
        end_event = th.cuda.Event(enable_timing = True)
        start_event.record()
        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = test_env.step(action)
            rewards_list.append(reward)
            action_return_tuples.append(info.get('action_return_tuple', None))
            if cum_reward is None:
                cum_reward = reward
            else:
                cum_reward += reward

            if done or truncated:
                end_event.record()
                th.cuda.synchronize()
                inference_time = start_event.elapsed_time(end_event)/1000 # in seconds
                # Render Change
                test_env.render()
                # env.render()
                time.sleep(1)


                route = info.get('route', None)
                if route is None:
                    print(f'Done {done}, truncated {truncated} Route is None for req_id {ep["req_id"]}, setting route to start node only')
                    route = [poiid2idx[ep['start_node']]]

                results_df.loc[idx] = [
                    ep['req_id'],
                    poiid2idx[ep['start_node']],
                    poiid2idx[ep['end_node']] if end_node_variant else None,
                    ep['time_constraint'],
                    route,
                    cum_reward,
                    rewards_list,
                    action_return_tuples,
                    inference_time
                ]
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'episode_results.csv'))
             
    #---------------------------------------------------------------------------------------#
    #%%
    # Compute and save test set metrics
    print('Compute and save test set metrics')

    REQ_ID_COL = 'req_id'
    ROUTE_COL = 'route'
    ROUTE_COL_CONTAINS_OSM_IDS = False
    REMOVE_START_NODE = True 
    GEOHASH_PRECISION = 7 # 153m × 153m	
    TEST_SET_W_CATEGORIES = not end_node_variant # whether test set contains user category prefs


    final_pois['geohash'] = final_pois['plotting_coords'].apply(
        lambda x: pgh.encode(x[0], x[1], precision = GEOHASH_PRECISION) # precision 6: 1.22km×0.61km
    )

    if CITY != 'verona':
        # convert visit duration from seconds to hours and insert into final_pois_df
        final_pois['min_visit_duration'] = final_pois['tourism_category'].apply(
            lambda x: min([
                NEW_VISIT_DURATION_BASED_ON_CATEGORIES[FINAL_CATEGORIES[idx]][0]/3600 for idx, val in enumerate(x) if val == 1
            ])
        )
        final_pois['max_visit_duration'] = final_pois['tourism_category'].apply(
            lambda x: max([
                NEW_VISIT_DURATION_BASED_ON_CATEGORIES[FINAL_CATEGORIES[idx]][1]/3600 for idx, val in enumerate(x) if val == 1
            ])
        )

        small_timebudget_test_set = test_set[test_set['time_constraint'].isin([2, 3, 4])]
        medium_timebudget_test_set = test_set[test_set['time_constraint'].isin([5, 6, 7])]
        large_timebudget_test_set = test_set[test_set['time_constraint'].isin([8, 9, 10])]

        small_timebudget_results_df = results_df[results_df['time_constraint'].isin([2, 3, 4])]
        medium_timebudget_results_df = results_df[results_df['time_constraint'].isin([5, 6, 7])]
        large_timebudget_results_df = results_df[results_df['time_constraint'].isin([8, 9, 10])]
    else:
        # for verona dataset, visit duration is already in minutes in the 'Time_Visit' column
        final_pois['min_visit_duration'] = final_pois['Time_Visit'] / 60
        final_pois['max_visit_duration'] = final_pois['Time_Visit'] / 60

        small_timebudget_test_set = test_set[(test_set['time_constraint'] >= 2) & (test_set['time_constraint'] <= 4)]
        medium_timebudget_test_set = test_set[(test_set['time_constraint'] >= 5) & (test_set['time_constraint'] <= 7)]
        large_timebudget_test_set = test_set[(test_set['time_constraint'] >= 8) & (test_set['time_constraint'] <= 10)]

        small_timebudget_results_df = results_df[(results_df['time_constraint'] >= 2) & (results_df['time_constraint'] <= 4)]
        medium_timebudget_results_df = results_df[(results_df['time_constraint'] >= 5) & (results_df['time_constraint'] <= 7)]
        large_timebudget_results_df = results_df[(results_df['time_constraint'] >= 8) & (results_df['time_constraint'] <= 10)]

    # remove 'start_node' and 'time_constraint' column for results_df of my approach
    results_df = results_df.drop(columns = ['start_node', 'time_constraint'], axis = 1)
    small_timebudget_results_df = small_timebudget_results_df.drop(columns = ['start_node', 'time_constraint'], axis = 1)
    medium_timebudget_results_df = medium_timebudget_results_df.drop(columns = ['start_node', 'time_constraint'], axis = 1)
    large_timebudget_results_df = large_timebudget_results_df.drop(columns = ['start_node', 'time_constraint'], axis = 1)

    ##
    # Global metrics
    ##
    print('########### GLOBAL METRICS ################')
    variant = 'global'
    if results_df.shape[0] > 0:
        compute_metrics_simplified_with_end_node(
            CITY,
            EXP_NAME,
            variant,
            results_df,
            test_set,
            REQ_ID_COL,
            ROUTE_COL,
            ROUTE_COL_CONTAINS_OSM_IDS,
            final_pois,
            REMOVE_START_NODE,
            OUTPUT_DIR,
            TEST_SET_W_CATEGORIES,
            poiid2idx,
            unfiltered_distance_matrix,
            start_node_pois_within_radius
        )
    print('##########################################')
    
    print('########### 2h-4h METRICS ################')
    variant = '2h_to_4h'
    if small_timebudget_results_df.shape[0] > 0:
        compute_metrics_simplified_with_end_node(
            CITY,
            EXP_NAME,
            variant,
            small_timebudget_results_df,
            small_timebudget_test_set,
            REQ_ID_COL,
            ROUTE_COL,
            ROUTE_COL_CONTAINS_OSM_IDS,
            final_pois,
            REMOVE_START_NODE,
            OUTPUT_DIR,
            TEST_SET_W_CATEGORIES,
            poiid2idx,
            unfiltered_distance_matrix,
            start_node_pois_within_radius
        )
    print('##########################################')

    print('########### 5h-7h METRICS ################')
    variant = '5h_to_7h'
    if medium_timebudget_results_df.shape[0] > 0:
        compute_metrics_simplified_with_end_node(
            CITY,
            EXP_NAME,
            variant,
            medium_timebudget_results_df,
            medium_timebudget_test_set,
            REQ_ID_COL,
            ROUTE_COL,
            ROUTE_COL_CONTAINS_OSM_IDS,
            final_pois,
            REMOVE_START_NODE,
            OUTPUT_DIR,
            TEST_SET_W_CATEGORIES,
            poiid2idx,
            unfiltered_distance_matrix,
            start_node_pois_within_radius
        )
    print('##########################################')

    print('########### 8h-10h METRICS ################')
    variant = '8h_to_10h'
    if large_timebudget_results_df.shape[0] > 0:
        compute_metrics_simplified_with_end_node(
            CITY,
            EXP_NAME,
            variant,
            large_timebudget_results_df,
            large_timebudget_test_set,
            REQ_ID_COL,
            ROUTE_COL,
            ROUTE_COL_CONTAINS_OSM_IDS,
            final_pois,
            REMOVE_START_NODE,
            OUTPUT_DIR,
            TEST_SET_W_CATEGORIES,
            poiid2idx,
            unfiltered_distance_matrix,
            start_node_pois_within_radius
        )
    print('##########################################')






# %%
