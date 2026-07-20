#%%
import os
import copy
import ast
import time

from tqdm import tqdm
import pandas as pd
import networkx as nx
import numpy as np

from utils import read_data, rescale_inputs
from eval import compute_metrics_simplified_with_end_node

import pygeohash as pgh

from constants import FINAL_CATEGORIES
from dataset_generation.constants import NEW_VISIT_DURATION_BASED_ON_CATEGORIES

#%%
def greedy_path_builder(final_pois_df, distance_matrix, idx2poiid, iteration_POI_graph, request_params):
    """
    Constructs paths starting from start node in request_params, inserting the highest scoring neighbor
    while respecting a distance budget. Returns a list of nodes forming route.
    
    Args:
        distance_matrix : Distance matrix
        final_pois_df : DataFrame with rows for each node and columns for node attributes
        iteration_POI_graph: NetworkX graph with nodes matching distance_matrix and final_pois_df
        request_params: Dictionary with keys ['start_node', 'time_constraint', 'category_prefs']
    
    Returns:
        route: list of node_ids
    """

    walking_speed = request_params['walking_speed']
    start_node = request_params['start_node']
    time_constraint = request_params['time_constraint']
    req_cat_prefs = request_params['category_prefs']
    
    # walkable distance
    distance_constraint = 10 if time_constraint <= 2 else 15
    score_col = request_params.get('score_col', 'distance')
    
    path = [start_node] # path without osm IDS
    visited = set(path)
    unvisited_cat_prefs = request_params['category_prefs'].copy() # when unvisited_cat_prefs is a null vector then all categories have been visited at least once
    current_node = start_node
    total_distance = 0
    temporal_total_distance = 0

    while True:
        # neighbors which are not visited yet and belong to request cat prefs
        neighbors = [n for n in iteration_POI_graph.neighbors(current_node) if n not in visited and req_cat_prefs.dot(final_pois_df[final_pois_df['_osm_id'] == n]['tourism_category'].values[0]) > 0]
        if not neighbors:
            break

        # Sort neighbors by score column
        if score_col == 'distance':
            neighbors = sorted(
                neighbors, key = lambda n: distance_matrix[current_node][n], reverse = False
            )
        else:
            neighbors = sorted(
                neighbors, key = lambda n: final_pois_df[final_pois_df['_osm_id'] == n][score_col].values[0], reverse = True
            )

        # Add nearest neighbor from unvisited_cat_prefs
        nearest_unvisited_cat_pref_neighbor = [n for n in neighbors if unvisited_cat_prefs.dot(final_pois_df[final_pois_df['_osm_id'] == n]['tourism_category'].values[0]) > 0]
        nearest_neighbor = nearest_unvisited_cat_pref_neighbor[0] if len(nearest_unvisited_cat_pref_neighbor) > 0 else neighbors[0]
        dist_to_nearest = distance_matrix[current_node][nearest_neighbor]
        dist_back_to_start = distance_matrix[nearest_neighbor][start_node]
        projected_total = total_distance + dist_to_nearest + dist_back_to_start

        temporal_dist_to_nearest = (dist_to_nearest/walking_speed) + iteration_POI_graph.nodes[nearest_neighbor].get('min_visit_time', 0)
        temporal_dist_back_to_start = distance_matrix[nearest_neighbor][start_node]/walking_speed
        temporal_projected_total = temporal_total_distance + temporal_dist_to_nearest + temporal_dist_back_to_start

        if (projected_total <= distance_constraint) and (temporal_projected_total <= time_constraint):

            path.append(nearest_neighbor)
            visited.add(nearest_neighbor)
            for idx, catval in enumerate(final_pois_df[final_pois_df['_osm_id'] == nearest_neighbor]['tourism_category'].values[0]):
                if catval == 1:
                    unvisited_cat_prefs[idx] = 0

            total_distance += dist_to_nearest
            temporal_total_distance += temporal_dist_to_nearest
            current_node = nearest_neighbor

        else:
            break  # No neighbor can be added within budget

    # Add return to start
    path.append(start_node)

    # convert all path nodes ids to OSM IDS
    _path = [idx2poiid[n] for n in path]

    return _path


#%%

if __name__ == '__main__':
    #%%
    CITY = 'new york' # Specify city name here: 'berlin', 'bonn', 'hamburg', 'new york', 'tokyo', 'verona'
    POI_graph_threshold = 3000 # in meters
    EXP_NAME = f'greedy_pref_{CITY}_th{POI_graph_threshold}'
    DATA_DIR = f'./data/{CITY}'
    OUTPUT_DIR = f'./results/{EXP_NAME}' 
    TEST_SET_PATH = f'./data/{CITY}/saved_data/test_set.csv'
    os.makedirs(OUTPUT_DIR, exist_ok = True)

    # dataframe with route requests of test set
    test_set = pd.read_csv(
        TEST_SET_PATH
    )
    print(f'test_set.columns {test_set.columns}, test_set.shape {test_set.shape}')

    test_set['cat_prefs_binary'] = test_set['cat_prefs'].apply(lambda x: np.array([1 if i in x else 0 for i in FINAL_CATEGORIES], dtype = np.float32))

    # dataframe to store model results on test set
    results_df = pd.DataFrame(
        columns = ['req_id', 'start_node', 'time_constraint', 'route', 'inference_time']
    )

    (
        final_pois, Bonn_walking_network, start_node_pois_within_radius,
        idx2poiid, poiid2idx, unfiltered_distance_matrix, node_category_attr_dict,
        POI_graph, distance_matrix, bearing_matrix
    ) = read_data(DATA_DIR, POI_graph_threshold)

    POI_graph, final_pois, distance_matrix, unfiltered_distance_matrix, start_node_pois_within_radius, _ = rescale_inputs(
        POI_graph,
        final_pois,
        poiid2idx,
        distance_matrix,
        unfiltered_distance_matrix,
        start_node_pois_within_radius
    )
    #%%
    iteration_POI_graph = copy.deepcopy(POI_graph)
    # remove all start nodes from iteration_POI_graph graph, and then iterate over test set and insert each start node of a request and remove the start node on termination
    iteration_POI_graph.remove_nodes_from(start_node_pois_within_radius)
    print(f'iteration_POI_graph {iteration_POI_graph}')
    
    for idx, row in tqdm(test_set.iterrows(), total = len(test_set)):

        req_id = row['req_id']
        start_node = poiid2idx[row['start_node']]
        time_constraint = row['time_constraint']

        start_time = time.perf_counter() # Start timer

        # insert start node and its attributes and edges in iteration_POI_graph
        iteration_POI_graph.add_node(start_node, **POI_graph.nodes[start_node])
        iteration_POI_graph.add_edges_from((start_node, nbr, POI_graph[start_node][nbr]) for nbr in POI_graph.neighbors(start_node) if nbr not in start_node_pois_within_radius)

        route = greedy_path_builder(
            final_pois,
            unfiltered_distance_matrix,
            idx2poiid,
            iteration_POI_graph,
            {
                'walking_speed': 5,
                'start_node': start_node,
                'time_constraint': time_constraint,
                'category_prefs': row['cat_prefs_binary'],
                'score_col': 'distance'
            }
        )
        
        # End timer
        end_time = time.perf_counter()
        inference_time = end_time - start_time # in seconds
        
        results_df.loc[idx] = [
            req_id,
            start_node,
            time_constraint,
            route,
            inference_time
        ]

        # remove start node and its attributes and edges in iteration_POI_graph
        iteration_POI_graph.remove_node(start_node)

    #%%
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'episode_results.csv'))

    #%%
    # Compute and save test set metrics
    print('Compute and save test set metrics')
    REQ_ID_COL = 'req_id'
    ROUTE_COL = 'route'
    ROUTE_COL_CONTAINS_OSM_IDS = True # Change to True for Naive baselines
    REMOVE_START_NODE = True 
    GEOHASH_PRECISION = 7 # 153m × 153m	
    TEST_SET_W_CATEGORIES = True # whether test set contains user category prefs


    final_pois['geohash'] = final_pois['plotting_coords'].apply(
        lambda x: pgh.encode(x[0], x[1], precision = GEOHASH_PRECISION) # precision 6: 1.22km×0.61km
    )

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
