#%%
import os
import pickle
import json
import ast

import pandas as pd 
import numpy as np
from itertools import combinations
from scipy.spatial import distance as scp_dist
from functools import reduce
import operator

import pygeohash as pgh

from constants import FINAL_CATEGORIES, VERONA_DATASET_CATEGORIES
from dataset_generation.constants import NEW_VISIT_DURATION_BASED_ON_CATEGORIES


#%%
def total_intra_list_distance(vectors):
    packed = [int(''.join('1' if x else '0' for x in v), 2) for v in vectors]
    total = 0
    n = len(packed)
    n_cat = len(vectors[0])
    for i in range(n):
        for j in range(i+1, n):
            total += ((packed[i] ^ packed[j]).bit_count()/n_cat) # normalised hamming distance
    return total

def compute_ild(vectors):
    n = len(vectors)
    tot = total_intra_list_distance(vectors)
    # number of distinct pairs = n*(n-1)/2
    return tot / (n*(n-1)/2)

##### Average Dice Coefficient #####

def average_duplicate_dice_sets(
    df,
    set_col
):

    def dice(a, b):
        inter = len(a & b)
        total = len(a) + len(b)
        return 1.0 if total == 0 else 2 * inter / total

    scores = []
    for _, grp in df.groupby('orig_group'):
        sets = grp[set_col].tolist()
        pair_scores = [dice(set(a), set(b)) for a, b in combinations(sets, 2)]
        # print(f'pair_scores {pair_scores}')
        if pair_scores != []:
            scores.append(np.mean(pair_scores))
    return float(np.mean(scores))
#############

def compute_metrics_simplified_with_end_node(
        CITY,
        MODEL_NAME,
        VARIANT,
        results_df,
        test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        TEST_SET_W_CATEGORIES,
        poiid2idx,
        distance_matrix, # in KM
        start_node_pois_within_radius
    ):
    
    # merge test set with results dataframe
    merged_results_df = pd.merge(results_df, test_set, left_on = REQ_ID_COL, right_on = 'req_id', how = 'inner')
    print(f'merged_results_df.columns {merged_results_df.columns}, merged_results_df.shape {merged_results_df.shape}')

    # change type of route column values to list
    if type(merged_results_df[ROUTE_COL].loc[0]) != list:
        try:
            merged_results_df[ROUTE_COL] = merged_results_df[ROUTE_COL].apply(lambda x: ast.literal_eval(x.replace('np.int64(', '').replace(')', '')))
        except Exception as e:
            print(e)
    
    # if route_col does not contain osm ids then replace id with osm id
    if not ROUTE_COL_CONTAINS_OSM_IDS:
        merged_results_df[ROUTE_COL] = merged_results_df[ROUTE_COL].apply(
            lambda x: [final_pois_df[final_pois_df['_osm_id'] == _id]['osm_id'].values[0] for _id in x]
        )
    
    # insert end_node column in merged_results_df
    # Note: Generalise function for round trips
    if 'end_node_ids' not in merged_results_df.columns:
        merged_results_df['end_node'] = merged_results_df['start_node']
    else:
        merged_results_df['end_node'] = merged_results_df['end_node_ids'].apply(
            lambda x: final_pois_df[final_pois_df['_osm_id'] == x]['osm_id'].values[0]
        )
    
    # remove start node from routes
    if REMOVE_START_NODE:
        merged_results_df[ROUTE_COL] = merged_results_df[ROUTE_COL].apply(
            lambda x: x[1:] if len(x) > 1 else []
        )

    ##
    # Remove end node (if end node == end node in request)
    ##
    routes_wo_end_node = []
    for i, row in merged_results_df.iterrows():

        if row[ROUTE_COL] == []:
            routes_wo_end_node.append(row[ROUTE_COL])

        elif row[ROUTE_COL][-1] == int(row['end_node']):
           routes_wo_end_node.append(row[ROUTE_COL][:-1])

        else:
            routes_wo_end_node.append(row[ROUTE_COL])
    
    merged_results_df['routes_wo_end_node'] = routes_wo_end_node

    # For metrics comparing duplicate requests, add original request group ids
    duplicates_per_sample = 3
    merged_results_df['orig_group'] = merged_results_df.index // duplicates_per_sample

    metrics_dict = {
        'rbl': None,
        'fbl': None, # routes within walkable distance and time constraint
        'acps': None,
        'pc': None,
        'ards': None,
        'ilcd': None,
        'inference_time': None,
        'mean_route_length': None,
        'rbld': None, # mean reachability distance
        'r_time_budget': None, # remaining time budget
        'r_distance_budget': None,
        'route_distance': None,
        'route_duration': None,
        'total_visit_time': None,
        'total_walk_time': None,
        'transition_length': None
    }


    merged_results_df['route_pois_count'] = merged_results_df['routes_wo_end_node'].apply(
        lambda x: len(x)
    ) 
    merged_results_df['n_unique_pois'] = merged_results_df['routes_wo_end_node'].apply(
        lambda x: len(set(x))
    )

    # metrics_dict['mean_route_length'] = merged_results_df['route_pois_count'].mean()
    
    if 'inference_time' in merged_results_df.columns:
        metrics_dict['inference_time'] = merged_results_df['inference_time'].mean()

    ##
    # Reachability
    ##
    # set reachability to 1 for classical model
    if MODEL_NAME == 'classical_model' or 'orientbaseline' in MODEL_NAME:
        metrics_dict['rbl'] = 1
        metrics_dict['rbld'] = 0
    else:

        # in how many cases did the route end at end node
        metrics_dict['rbl'] = sum(
            merged_results_df.apply(
                lambda row: int(row['end_node']) == row[ROUTE_COL][-1] if len(row[ROUTE_COL]) > 0 else False,
                axis = 1
            )
        )/ len(merged_results_df)

        reachability_distance = []
        for i, row in merged_results_df.iterrows():

            if row[ROUTE_COL] == []:
                reachability_distance.append(None)

            elif row[ROUTE_COL][-1] == int(row['end_node']):
                reachability_distance.append(0)
            else:
                # get distance from start node to end node
                req_end_node_idx = poiid2idx[int(row['end_node'])]
                end_node_idx = poiid2idx[row[ROUTE_COL][-1]]
                distance = distance_matrix[req_end_node_idx][end_node_idx]
                reachability_distance.append(distance)
        merged_results_df['reachability_distance'] = reachability_distance

        metrics_dict['rbld'] = merged_results_df['reachability_distance'].mean()
    
    ##
    # Feasibility
    ##
    n_routes_within_walkable_distance = []
    n_routes_within_time_constraint = []
    route_durations = []
    total_visit_time = []
    total_walk_time = []
    route_lengths = []
    all_routes_transition_lengths = [] # over all route transitions
    remaining_time_budgets = []
    remaining_distance_budgets = []
    
    if MODEL_NAME == 'classical_model' or 'orientbaseline' in MODEL_NAME:

        for i, row in merged_results_df.iterrows():

            if len(row[ROUTE_COL]) != 0:
                
                # get distance from start node to end node
                start_node_idx = poiid2idx[int(row['start_node'])]
                end_node_idx = poiid2idx[int(row['end_node'])]

                first_poi_idx = poiid2idx[row[ROUTE_COL][0]]

                route_distance = distance_matrix[start_node_idx][first_poi_idx]
                transition_lengths= [route_distance]

                visit_time = 0
                for idx in range(len(row[ROUTE_COL]) - 1):
                    route_distance += distance_matrix[poiid2idx[row[ROUTE_COL][idx]]][poiid2idx[row[ROUTE_COL][idx + 1]]]
                    transition_lengths.append(distance_matrix[poiid2idx[row[ROUTE_COL][idx]]][poiid2idx[row[ROUTE_COL][idx + 1]]])
                    visit_time += final_pois_df[final_pois_df['osm_id'] == row[ROUTE_COL][idx]]['min_visit_duration'].values[0]
                
                total_visit_time.append(visit_time)
                total_walk_time.append(route_distance/5)
                route_time = visit_time + (route_distance/5) # 5km/h walking speed
                route_durations.append(route_time)
                route_lengths.append(route_distance)
                all_routes_transition_lengths.append(sum(transition_lengths)/len(transition_lengths)) 
                

                time_constraint = row['time_constraint']
                walkable_threshold = 10 if time_constraint == 2 else 15 # in km
                n_routes_within_walkable_distance.append(
                    False if route_distance > walkable_threshold else True
                )
                n_routes_within_time_constraint.append(
                    False if round(route_time - time_constraint, 1) > 0.1 else True
                )
                # remaining time budget
                remaining_time = time_constraint - route_time
                remaining_time_budgets.append(remaining_time)

                # remaining distance budget
                remaining_distance = walkable_threshold - route_distance
                remaining_distance_budgets.append(remaining_distance)
                

            else:
                n_routes_within_walkable_distance.append(False)
                n_routes_within_time_constraint.append(False)
                route_durations.append(None)
                total_visit_time.append(None)
                total_walk_time.append(None)
                route_lengths.append(None)
                all_routes_transition_lengths.append(None)
                remaining_time_budgets.append(None)
                remaining_distance_budgets.append(None)
    else:       

        for i, row in merged_results_df.iterrows():
            if row[ROUTE_COL] == []:
                n_routes_within_walkable_distance.append(False)
                n_routes_within_time_constraint.append(False)
                route_durations.append(None)
                total_visit_time.append(None)
                total_walk_time.append(None)
                route_lengths.append(None)
                all_routes_transition_lengths.append(None)
                remaining_time_budgets.append(None)
                remaining_distance_budgets.append(None)

            else:
                # get distance from start node to end node
                start_node_idx = poiid2idx[int(row['start_node'])]
                end_node_idx = poiid2idx[int(row['end_node'])]
                first_poi_idx = poiid2idx[row[ROUTE_COL][0]]

                route_distance = distance_matrix[start_node_idx][first_poi_idx]
                transition_lengths= [route_distance]

                visit_time = 0
                for idx in range(len(row[ROUTE_COL]) - 1):
                    route_distance += distance_matrix[poiid2idx[row[ROUTE_COL][idx]]][poiid2idx[row[ROUTE_COL][idx + 1]]]
                    transition_lengths.append(distance_matrix[poiid2idx[row[ROUTE_COL][idx]]][poiid2idx[row[ROUTE_COL][idx + 1]]])
                    visit_time += final_pois_df[final_pois_df['osm_id'] == row[ROUTE_COL][idx]]['min_visit_duration'].values[0]
                
                ##
                # Note: Commented out below two lines as distance to end node is computed in above loop
                ##
                # visit_time += final_pois_df[final_pois_df['osm_id'] == row[ROUTE_COL][-1]]['min_visit_duration'].values[0]
                # route_distance += distance_matrix[poiid2idx[row[ROUTE_COL][-1]]][end_node_idx]
                total_visit_time.append(visit_time)
                total_walk_time.append(route_distance/5)
                route_time = visit_time + (route_distance/5) # 5km/h walking speed
                route_durations.append(route_time)
                route_lengths.append(route_distance)
                all_routes_transition_lengths.append(sum(transition_lengths)/len(transition_lengths)) 
                # route_distance = route_distance # convert to km

                time_constraint = row['time_constraint']
                walkable_threshold = 10 if time_constraint == 2 else 15 # in km
                n_routes_within_walkable_distance.append(
                    False if route_distance > walkable_threshold else True
                )
                n_routes_within_time_constraint.append(
                    False if round(route_time - time_constraint, 1) > 0.1 else True
                    # False if (round(route_time, 2) > round(time_constraint, 2)) else True
                )
                # remaining time budget
                remaining_time = time_constraint - route_time
                remaining_time_budgets.append(remaining_time)

                # remaining distance budget
                remaining_distance = walkable_threshold - route_distance
                remaining_distance_budgets.append(remaining_distance)
            
    merged_results_df['within_walkable_distance'] = n_routes_within_walkable_distance
    merged_results_df['within_time_constraint'] = n_routes_within_time_constraint
    merged_results_df['route_durations'] = route_durations
    merged_results_df['total_visit_time'] = total_visit_time
    merged_results_df['total_walk_time'] = total_walk_time
    merged_results_df['route_lengths'] = route_lengths
    merged_results_df['remaining_time_budget'] = remaining_time_budgets
    merged_results_df['remaining_time_budget'] = merged_results_df['remaining_time_budget'].round(3)
    merged_results_df['remaining_distance_budget'] = remaining_distance_budgets
    merged_results_df['all_routes_transition_lengths'] = all_routes_transition_lengths
    merged_results_df['feasible'] = merged_results_df['within_walkable_distance'] & merged_results_df['within_time_constraint']

    metrics_dict['fbl'] = merged_results_df['feasible'].sum()/len(merged_results_df['feasible'])

    if CITY == 'verona':
        poi_list = list(final_pois_df['osm_id'].values)
    else:
        poi_list = list(final_pois_df[~final_pois_df['_osm_id'].isin(start_node_pois_within_radius)]['osm_id'].values)

    merged_results_df.to_csv(
        os.path.join(METRICS_SAVE_PATH, f'{VARIANT}_intermediate_results.csv'),
        index = False
    )
    ##
    # Remaining metrics computed over feasible routes only
    ##
    merged_results_df = merged_results_df[merged_results_df['feasible'] == True]
    print(f'Number of feasible routes: {len(merged_results_df)}')
    
    ##
    # Route Distance & Duration Metrics
    ##
    metrics_dict['mean_route_length'] = merged_results_df['route_pois_count'].mean()
    metrics_dict['route_distance'] = merged_results_df['route_lengths'].mean()
    metrics_dict['route_duration'] = merged_results_df['route_durations'].mean()
    metrics_dict['total_visit_time'] = merged_results_df['total_visit_time'].mean()
    metrics_dict['total_walk_time'] = merged_results_df['total_walk_time'].mean()
    metrics_dict['r_distance_budget'] = merged_results_df['remaining_distance_budget'].mean()
    metrics_dict['r_time_budget'] = merged_results_df['remaining_time_budget'].mean()
    metrics_dict['transition_length'] = merged_results_df['all_routes_transition_lengths'].mean()

    # Standard Deviations
    metrics_dict['std_mean_route_length'] = merged_results_df['route_pois_count'].std()
    metrics_dict['std_route_distance'] = merged_results_df['route_lengths'].std()
    metrics_dict['std_route_duration'] = merged_results_df['route_durations'].std()
    metrics_dict['std_total_visit_time'] = merged_results_df['total_visit_time'].std()
    metrics_dict['std_total_walk_time'] = merged_results_df['total_walk_time'].std()
    metrics_dict['std_r_distance_budget'] = merged_results_df['remaining_distance_budget'].std()
    metrics_dict['std_r_time_budget'] = merged_results_df['remaining_time_budget'].std()
    metrics_dict['std_transition_length'] = merged_results_df['all_routes_transition_lengths'].std()


    ##
    # POI Coverage
    ##
    route_pois = set(poi for route in list(merged_results_df['routes_wo_end_node'].values) for poi in route)
    total_pois = len(poi_list)
    metrics_dict['pc'] = len(route_pois)/total_pois

    if CITY == 'verona':
        merged_results_df['route_w_categories'] =  merged_results_df['routes_wo_end_node'].apply(
            lambda x: [
                final_pois_df[final_pois_df['osm_id'] == osm_id]['tourism_category'].values[0] for osm_id in x
            ]
        )
    else:
        merged_results_df['route_w_categories'] =  merged_results_df['routes_wo_end_node'].apply(
            lambda x: [
                final_pois_df[final_pois_df['osm_id'] == osm_id]['tourism_category'].values[0][4:] for osm_id in x
            ]
        )

    ##   
    # Intra-list Diversity
    ##
    ild_result_list = []
    for route in merged_results_df['route_w_categories']:
        # pdb.set_trace()
        # ILD needs at least 2 POIs to compute diversity
        if len(route) < 2:
            ild_result_list.append(0)
            continue
        ild_result = compute_ild(route)
        # normalise and save ild
        if CITY == 'verona':
            ild_result_list.append(ild_result/len(VERONA_DATASET_CATEGORIES))
        else:
            ild_result_list.append(ild_result/len(FINAL_CATEGORIES[4:]))
    
    metrics_dict['ilcd'] = np.mean(ild_result_list) if len(ild_result_list) > 0 else 0
    print(f'metrics_dict[ilcd] {metrics_dict["ilcd"]}')


    ##
    # Average Dice Coefficent
    ##
    avg_dice_coeff = average_duplicate_dice_sets(
        merged_results_df,
        set_col = 'routes_wo_end_node'
    )
    metrics_dict['ards'] = avg_dice_coeff if len(merged_results_df) > 0 else 1.0
    print(f'metrics_dict[ards] {metrics_dict["ards"]}')

    ##
    # ACPS
    ##
    ## Category preference dice similarity
    if TEST_SET_W_CATEGORIES:
        cat_pref_sim_results = []

        if CITY == 'verona':
            merged_results_df['cat_prefs_binary'] = merged_results_df['cat_prefs'].apply(lambda x: np.array([1 if i in x else 0 for i in VERONA_DATASET_CATEGORIES], dtype = np.float32))
        else:
            merged_results_df['cat_prefs_binary'] = merged_results_df['cat_prefs'].apply(lambda x: np.array([1 if i in x else 0 for i in FINAL_CATEGORIES[4:]], dtype = np.float32))
        for idx, row in merged_results_df.iterrows():

            similarity = 0
            if len(row['route_w_categories']) > 0:

                route_cats = reduce(operator.or_, map(np.array, row['route_w_categories']))
                request_cats = row['cat_prefs_binary']
                similarity = 1 - scp_dist.dice(route_cats, request_cats)

            cat_pref_sim_results.append(similarity)

        metrics_dict['acps'] = float(np.mean(cat_pref_sim_results)) if len(cat_pref_sim_results) > 0 else 0


    print(f'metrics_dict {metrics_dict}')
    with open(os.path.join(METRICS_SAVE_PATH, f'{MODEL_NAME}_rep_metrics.txt'), 'a') as f:
        f.write(VARIANT)
        json.dump(metrics_dict, f, indent = 4)

    # save metrics dict to pickle file
    with open(os.path.join(METRICS_SAVE_PATH, f'{MODEL_NAME}_{VARIANT}_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_dict, f)

    # merge results into existing metrics csv file
    metrics_dict['variant'] = VARIANT

    if os.path.exists(os.path.join(METRICS_SAVE_PATH, f'{MODEL_NAME}_rep_metrics.csv')):
        existing_metrics_df = pd.read_csv(os.path.join(METRICS_SAVE_PATH, f'{MODEL_NAME}_rep_metrics.csv'))
        print(f'existing_metrics_df.columns {existing_metrics_df.columns}, existing_metrics_df.shape {existing_metrics_df.shape}')
        if VARIANT in existing_metrics_df['variant'].values:
            print(f'Variant {VARIANT} already exists in metrics csv file. Overwriting the existing entry.')
            existing_metrics_df = existing_metrics_df[existing_metrics_df['variant'] != VARIANT]
        
        new_metrics_df = pd.DataFrame([metrics_dict])
        df_with_merged_metrics = pd.concat([existing_metrics_df, new_metrics_df], ignore_index = True)
    else:
        df_with_merged_metrics = pd.DataFrame([metrics_dict])
    # save merged results dataframe to csv
    df_with_merged_metrics.to_csv(
        os.path.join(METRICS_SAVE_PATH, f'{MODEL_NAME}_rep_metrics.csv'),
        index = False
    )

#############
#%%
if __name__ == '__main__':
    #%%
    CITY = 'verona'
    DATA_PATH = f'./data/{CITY}/saved_data'
    
    # path to model results dataframe and metrics save path
    MODEL_NAME = f'exp_{CITY}'
    RESULTS_PATH = f'./results/{MODEL_NAME}/episode_results.csv' 

    TEST_SET_PATH = f'./data/{CITY}/saved_data/test_set.csv'
    METRICS_SAVE_PATH = f'./results/{MODEL_NAME}/metrics'
    METRICS_RESULTS_CSV_PATH = METRICS_SAVE_PATH 
    os.makedirs(METRICS_SAVE_PATH, exist_ok = True)
    os.makedirs(METRICS_RESULTS_CSV_PATH, exist_ok = True)
    
    # model specific column names
    IS_CLASSICAL_MODEL = True # whether model is a classical model i.e orienteering baseline
    IS_RL_BASELINE = False # whether model is RL baseline
    REQ_ID_COL = 'req_id'
    ROUTE_COL = 'Cycle_w_osm_ids' #'itinerary' #'Cycle_w_osm_ids' #'route'
    ROUTE_COL_CONTAINS_OSM_IDS = True # Change to True for Naive baselines
    REMOVE_START_NODE = True 
    GEOHASH_PRECISION = 7 # 153m × 153m	
    TEST_SET_W_CATEGORIES = True if CITY != 'verona' else False # whether test set contains user category prefs
    #%%

    # load poi data, distance matrices, start nodes
    with open(f'{DATA_PATH}/indexing_dicts.pkl', 'rb') as f:
        indexing_dicts = pickle.load(f)
    print(f'indexing_dicts {len(indexing_dicts)}')
    idx2poiid = indexing_dicts['idx2poiid']
    poiid2idx = indexing_dicts['poiid2idx']
    print(f'len(idx2poiid) {len(idx2poiid)}, len(poiid2idx) {len(poiid2idx)}')

    final_pois_df = pd.read_csv(f'{DATA_PATH}/final_pois.csv')
    print(final_pois_df.shape)
    #%%

    if CITY == 'verona':
        final_pois_df['tourism_category'] = final_pois_df['tourism_category'].apply(
            lambda x:
            [
                int(i) for i in x.strip('[.]').split(', ')
            ]
        )

        final_pois_df['plotting_coords'] = final_pois_df['plotting_coords'].apply(
            lambda x: x.replace('np.float64', '').replace('), (', ', ').strip('()')
        )

    else:
        final_pois_df['tourism_category'] = final_pois_df['tourism_category'].apply(
            lambda x:
            [
                int(i) for i in x.strip('[.]').split('. ')
            ]
        )
    final_pois_df['plotting_coords'] = final_pois_df['plotting_coords'].apply(
        lambda x: ast.literal_eval(x)
    )

    final_pois_df['geohash'] = final_pois_df['plotting_coords'].apply(
        lambda x: pgh.encode(x[0], x[1], precision = GEOHASH_PRECISION) # precision 6: 1.22km×0.61km
    )


    with open(f'{DATA_PATH}/start_node_pois_within_radius.pkl', 'rb') as f:
        start_node_pois_within_radius = pickle.load(f)
    start_node_pois_within_radius = [poiid2idx[_id] for _id in start_node_pois_within_radius]
    print(f'len(start_node_pois_within_radius) {len(start_node_pois_within_radius)}')
    #%%
    # Distance matrix in meters
    with open(f'{DATA_PATH}/distance_matrix.npy', 'rb') as f:
        distance_matrix = np.load(f)
    print(f'distance_matrix {distance_matrix.shape}')
    # Convert distance matrix to KM
    distance_matrix = distance_matrix / 1000
        
    # convert visit duration from seconds to hours and insert into final_pois_df
    if CITY != 'verona':
        final_pois_df['min_visit_duration'] = final_pois_df['tourism_category'].apply(
            lambda x: min([
                NEW_VISIT_DURATION_BASED_ON_CATEGORIES[FINAL_CATEGORIES[idx]][0]/3600 for idx, val in enumerate(x) if val == 1
            ])
        )
        final_pois_df['max_visit_duration'] = final_pois_df['tourism_category'].apply(
            lambda x: max([
                NEW_VISIT_DURATION_BASED_ON_CATEGORIES[FINAL_CATEGORIES[idx]][1]/3600 for idx, val in enumerate(x) if val == 1
            ])
        )
    else:
        # for verona dataset, visit duration is already in minutes in the 'Time_Visit' column
        final_pois_df['min_visit_duration'] = final_pois_df['Time_Visit'] / 60
        final_pois_df['max_visit_duration'] = final_pois_df['Time_Visit'] / 60
    #%%
    # dataframe with route requests of test set
    test_set = pd.read_csv(
        TEST_SET_PATH
    )
    small_timebudget_test_set = test_set[(test_set['time_constraint'] >= 2) & (test_set['time_constraint'] <= 4)]
    medium_timebudget_test_set = test_set[(test_set['time_constraint'] >= 5) & (test_set['time_constraint'] <= 7)]
    large_timebudget_test_set = test_set[(test_set['time_constraint'] >= 8) & (test_set['time_constraint'] <= 10)]
    print(f'test_set.columns {test_set.columns}, test_set.shape {test_set.shape}, small_timebudget_test_set.shape {small_timebudget_test_set.shape}, medium_timebudget_test_set.shape {medium_timebudget_test_set.shape}, large_timebudget_test_set.shape {large_timebudget_test_set.shape} ')

    results_df = pd.read_csv(
        RESULTS_PATH
    )
    # insert time_constraint in classical model results
    if IS_CLASSICAL_MODEL or IS_RL_BASELINE:
        results_df['time_constraint'] = results_df[REQ_ID_COL].apply(
            lambda rid: test_set[test_set['req_id'] == rid]['time_constraint'].iloc[0]
        )
        results_df['start_node'] = results_df[REQ_ID_COL].apply(
            lambda rid: test_set[test_set['req_id'] == rid]['start_node'].iloc[0]
        )
    small_timebudget_results_df = results_df[(results_df['time_constraint'] >= 2) & (results_df['time_constraint'] <= 4)]
    medium_timebudget_results_df = results_df[(results_df['time_constraint'] >= 5) & (results_df['time_constraint'] <= 7)]
    large_timebudget_results_df = results_df[(results_df['time_constraint'] >= 8) & (results_df['time_constraint'] <= 10)]
    print(f'results_df.columns {results_df.columns}, results_df.shape {results_df.shape}, small_timebudget_results_df.shape {small_timebudget_results_df.shape}, medium_timebudget_results_df.shape {medium_timebudget_results_df.shape}, large_timebudget_results_df.shape {large_timebudget_results_df.shape}')
   
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
        MODEL_NAME,
        variant,
        results_df,
        test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        TEST_SET_W_CATEGORIES,
        poiid2idx,
        distance_matrix,
        start_node_pois_within_radius
    )
    print('##########################################')
    
    print('########### 2h-4h METRICS ################')
    variant = '2h_to_4h'
    compute_metrics_simplified_with_end_node(
        CITY,
        MODEL_NAME,
        variant,
        small_timebudget_results_df,
        small_timebudget_test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        TEST_SET_W_CATEGORIES,
        poiid2idx,
        distance_matrix,
        start_node_pois_within_radius
    )
    print('##########################################')

    print('########### 5h-7h METRICS ################')
    variant = '5h_to_7h'
    compute_metrics_simplified_with_end_node(
        CITY,
        MODEL_NAME,
        variant,
        medium_timebudget_results_df,
        medium_timebudget_test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        TEST_SET_W_CATEGORIES,
        poiid2idx,
        distance_matrix,
        start_node_pois_within_radius
    )
    print('##########################################')

    print('########### 8h-10h METRICS ################')
    variant = '8h_to_10h'
    compute_metrics_simplified_with_end_node(
        CITY,
        MODEL_NAME,
        variant,
        large_timebudget_results_df,
        large_timebudget_test_set,
        REQ_ID_COL,
        ROUTE_COL,
        ROUTE_COL_CONTAINS_OSM_IDS,
        final_pois_df,
        REMOVE_START_NODE,
        METRICS_SAVE_PATH,
        TEST_SET_W_CATEGORIES,
        poiid2idx,
        distance_matrix,
        start_node_pois_within_radius
    )
    print('##########################################')

#############
# %%
