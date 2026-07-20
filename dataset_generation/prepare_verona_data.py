'''
Prepare Verona dataset as follows:
1. POIs only consist of Verona dataset POIs
2. Visit time, POI popularity set from Verona dataset
3. Train, Validation and Test set contain start, end and time budget
and is created using Verona dataset routes from year 2018 - 2019
4. Road Network and Distance matrices from OSMnx
'''

#%%
import pandas as pd
import numpy as np
from itertools import islice
import random

import osmnx as ox
import networkx as nx
import pickle

import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.model_selection import train_test_split
from shapely.geometry import Point

from enum import Enum
from haversine import haversine, Unit

from tqdm import tqdm

import folium
import requests
from shapely import wkt
from math import isnan
import time
import os
import copy

from scipy.spatial import distance
import seaborn as sns

from collections import Counter

from constants import (
    VERONA_DATASET_CATEGORIES
)

VERONA_DATASET_CATEGORIES2IDS = {
    'Musei e Centri Espositivi': 0,
    'Monumenti': 1,
    'Chiese': 2
}


#%% 

def distance_matrix_to_poi_graph(
        distance_matrix,
        threshold = 3000,
        node_min_visit_time_attr_dict = None,
        node_max_visit_time_attr_dict = None
    ):
    
    updated_distance_matrix = copy.deepcopy(distance_matrix)
    updated_distance_matrix[updated_distance_matrix > threshold] = np.inf
    
    # construct POI graph based on new distance matrix
    POI_graph = nx.Graph()
    # insert edges in POI_graph if distance is non-zero and not np.inf in distance_matrix
    for i, source in idx2poiid.items():
        POI_graph.add_node(source) # add all nodes first
        for j, target in idx2poiid.items():
            if i != j and updated_distance_matrix[i, j] != np.inf:
                POI_graph.add_edge(source, target, weight = updated_distance_matrix[i, j])
    
    print(f'POI_graph.number_of_nodes(), POI_graph.number_of_edges() {POI_graph.number_of_nodes(), POI_graph.number_of_edges()}')
    print(f'Min and max degree {min([x[1] for x in list(POI_graph.degree())]), max([x[1] for x in list(POI_graph.degree())])}')
    print(f'Min Max edge weight {max(data['weight'] for u, v, data in POI_graph.edges(data=True)), min(data['weight'] for u, v, data in POI_graph.edges(data=True))}')

    nx.set_node_attributes(POI_graph, node_min_visit_time_attr_dict, 'min_visit_time')
    nx.set_node_attributes(POI_graph, node_max_visit_time_attr_dict, 'max_visit_time')

    return POI_graph, updated_distance_matrix


def prepare_bearing_matrix(poiid2idx, poi_data):

    # initialise bearing_matrix with zeros
    bearing_matrix = np.zeros((len(poiid2idx), len(poiid2idx)))

    # Pre-extract coordinates into a dictionary (O(n) one-time cost)
    coords_dict = {}
    for osm_id, coords in zip(poi_data['osm_id'], poi_data['plotting_coords']):
        coords_dict[osm_id] = coords
    
    poi_ids = list(poiid2idx.keys())
    n_pois = len(poi_ids)
    for i, source in tqdm(enumerate(poi_ids), total=n_pois):

        lat1, lon1 = coords_dict[source]

        for j, target in enumerate(poi_ids):

            if i >= j:  # Compute bearing only for upper triangle (including diagonal)
                continue
           
            lat2, lon2 = coords_dict[target]

            # Get bearing (in decimal degrees) from the source to all other nodes; (lat1, lon1, lat2, lon2)
            bearing = ox.bearing.calculate_bearing(
                lat1, lon1, lat2, lon2
            )
            bearing_matrix[poiid2idx[source], poiid2idx[target]] = bearing

            # Since bearing from A to B is 180 degrees opposite to bearing from B to A, we can compute the bearing for the upper triangle using the bearing for the lower triangle
            bearing_matrix[poiid2idx[target], poiid2idx[source]] = (bearing + 180) % 360

    
    print(f'bearing_matrix.shape {bearing_matrix.shape}')
    print(f'np.count_nonzero(bearing_matrix) {np.count_nonzero(bearing_matrix)}')
    
    return bearing_matrix

def poi_transition_matrix(df, item_col = None, item_count = None):

    # create item_count x item_count item transition matrix
    transition_count_matrix = np.zeros((item_count, item_count))
    
    for traj in df[item_col]:
        if len(traj) < 2:
            continue
        

        for src, dst in zip(traj[:-1], traj[1:]):
            if src == dst:
                continue

            transition_count_matrix[src, dst] = transition_count_matrix[src, dst] + 1
    

    # Row-normalize to probabilities
    row_sums = transition_count_matrix.sum(axis = 1, keepdims = True)
    
    zero_rows = (row_sums[:, 0] == 0)
    row_sums[row_sums == 0] = 1 # to avoid division by zero error
    transition_prob_matrix = transition_count_matrix / row_sums
    # adding self loops for zero prob rows to create a valid markov chain
    transition_prob_matrix[zero_rows, zero_rows] = 1.0

    print(f'Zero rows {zero_rows}')

    return transition_count_matrix, transition_prob_matrix

def verona_df_to_request_df(
        df_poi,
        poi_data,
        distance_matrix,
        poiid2idx,
        MIN_TC_REQUESTS = None,
        RANDOM_SEED = 0,
        MAX_TRAIN_REQUESTS = None
    ):
    '''
    Convert Verona dataset dataframe to request dataframe with start_node, end_node, time constraint, route_pois.
    '''

    df_request = pd.DataFrame(
        columns = ['req_id', 'start_node', 'end_node', 'time_constraint', 'route_pois', 'route_length']
    )
    
    # Group by user and visit date to get individual routes and only keep routes with more than 1 POI 
    # excluding start and end location
    grouped_df = df_poi.groupby(
        ['id_veronacard', 'data_visita']
    ).filter(lambda g: g.shape[0] > 2 and len(set(g['poi'])) > 2)

    grouped_df = grouped_df.groupby(['id_veronacard', 'data_visita'])
    print(f'Number of unique requests in df_poi: {len(grouped_df)}')

    if MAX_TRAIN_REQUESTS is not None:
        grouped_df = list(islice(grouped_df, MAX_TRAIN_REQUESTS))
        print(f'Sampled requests: {len(grouped_df)}')

    req_id = 0
    for _, group_data in tqdm(list(grouped_df), total = len(grouped_df)):
        
        start_node = group_data['poi'].iloc[0]
        end_node = group_data['poi'].iloc[-1]
        
        # create route pois after dropping duplicate pois (except same start and end node)
        seen = set()
        route_pois = [p for p in group_data['poi'].tolist() if not (p in seen or seen.add(p))]

        # calculate time constraint as sum of visit time and travel time between consecutive pois in route
        # also computing route length
        time_constraint = 0
        route_length = 0
        for i in range(len(route_pois) - 1):
            poi_start = route_pois[i]
            poi_dest = route_pois[i + 1]

            # visit time in hours
            visit_time = 0
            if i != 0: # Excluding visit time of start location
                visit_time = poi_data[poi_data['id'] == poi_start]['Time_Visit'].values[0] / 60

            # travel time in hours (convert distance to time using walking speed of 5km/h)
            travel_time = distance_matrix[
                poiid2idx[poi_start], poiid2idx[poi_dest]
            ] / (5*1000)
            time_constraint += visit_time + travel_time

            # walking distance in meters
            route_length += distance_matrix[
                poiid2idx[poi_start], poiid2idx[poi_dest]
            ]
        
        df_request.loc[req_id] = [
            req_id,
            start_node,
            end_node,
            time_constraint,
            route_pois,
            route_length
        ]
        req_id += 1

    df_request['rounded_time_constraint'] = df_request['time_constraint'].round()
    
    if MIN_TC_REQUESTS is not None:
        # create a stratified train val split based on time constraint
        tc_dfs = []
        for tc in range(2, 11):

            _df = df_request[
                (df_request['rounded_time_constraint'] == tc )
            ]
            min_df_shape = min(_df.shape[0], MIN_TC_REQUESTS)
            tc_dfs.append(_df.sample(n = min_df_shape, random_state = RANDOM_SEED))

        df_request = pd.concat(tc_dfs).sample(frac = 1, random_state = RANDOM_SEED).reset_index(drop=True) # shuffle the concatenated dataframe

        # renumber requests
        df_request['req_id'] = range(0, len(df_request))
            
    print(f'df_request.shape {df_request.shape}')
    
    return df_request

#%%
if __name__ == '__main__':
    #%%
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)
    
    #%%
    duplicate_copies = 3 # Number of duplicate requests for each request in test set
    CITY = 'verona'
    MAX_TRAIN_REQUESTS = None # these requests will be further sampled based on time constraint

    OSM_ID = '<verona_osm_id>' # Verona
    CITY_LAT_LON = '<verona_city_center_lat_lon>' # Verona
    DATASET_PATH = f'../data/<verona_dataset_path>' # Provide path to Veroan dataset from Vecchia et al.
    SAVE_PATH = f'../data/{CITY}/saved_data'
    os.makedirs(SAVE_PATH, exist_ok = True)

    #%%
    ##
    # POI dataframe with id, category, coordinates and visit time
    ##
    ## Since Verona dataset has 14 POIs in 2018 and 2019
    poi_data = pd.read_csv(os.path.join(DATASET_PATH, 'poi_it.csv'))[:14]
    print(f'poi_data.shape {poi_data.shape}')

    poi_data = gpd.GeoDataFrame(
        poi_data, geometry = gpd.points_from_xy(poi_data.longitude, poi_data.latitude), crs = 'EPSG:4326'
    )

    # dummy osm id
    poi_data['osm_id'] = poi_data['id']

    # indexing POIs 
    idx2poiid = {}
    poiid2idx = {}
    for idx, poi_id in enumerate(poi_data['id']):
        idx2poiid[idx] = poi_id
        poiid2idx[poi_id] = idx
    print(f'len(idx2poiid), len(poiid2idx) {len(idx2poiid), len(poiid2idx)}')
    
    # merge poi popularity with poi data
    poi_popularity = pd.read_csv(os.path.join(DATASET_PATH, 'poi_popularity_2023.csv'))
    print(f'poi_popularity {poi_popularity.shape}')
    poi_data['popularity'] = poi_data['id'].apply(
        lambda x: poi_popularity[poi_popularity['poi'] == x]['popularity'].values[0]
    )
    
    poi_data['importance_score'] = poi_data['popularity']
    start_node_poi_ids = list(idx2poiid.values()) # all pois are start node pois in Verona dataset
    n_pois = len(poiid2idx)

    #%%
    weather_df_names = ['weather_2022_processed.csv', 'weather_2023_processed.csv']
    weather_dfs = []
    for n in weather_df_names:
        weather_dfs.append(
            pd.read_csv(os.path.join(DATASET_PATH, n))
        )
    weather_df = pd.concat(weather_dfs)
    weather_df = weather_df.set_index('date')
    print(f'weather_df.shape {weather_df.shape}')
    #%%
    # bearing_matrix
    # (lat, lon)
    poi_data['plotting_coords'] = poi_data[['latitude', 'longitude']].apply(
        lambda row: (row['latitude'], row['longitude']),
        axis = 1
    )
    bearing_matrix = prepare_bearing_matrix(poiid2idx, poi_data)

    #%%
    coords = []
    for idx, poiid in idx2poiid.items():
        coords.append(
            poi_data[poi_data['id'] == poiid]['plotting_coords'].values[0]
        )


    # Download walking network around the points
    center_lat = np.mean([c[0] for c in coords])
    center_lon = np.mean([c[1] for c in coords])

    G = ox.graph_from_point(
        (center_lat, center_lon),
        dist = 10000,# meters
        network_type = 'walk'
    )
    #%%
    # Snap coordinates to nearest nodes
    nodes = [
        ox.distance.nearest_nodes(G, X = lon, Y = lat)
        for lat, lon in coords
    ]

    # Distance matrix (km)
    n = len(nodes)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(
            G,
            nodes[i],
            weight = 'length'
        )

        for j in range(n):
            distance_matrix[i, j] = lengths.get(nodes[j], np.inf)


    # node attribute dicts - saveing visiting time in seconds as node attributes in POI graph
    node_min_visit_time_attr_dict = dict(zip(poi_data['id'], poi_data['Time_Visit'] * 60))
    node_max_visit_time_attr_dict = dict(zip(poi_data['id'], poi_data['Time_Visit'] * 60))
    category_list = np.zeros((len(VERONA_DATASET_CATEGORIES),))
    poi_data['tourism_category'] = poi_data.apply(
        lambda row: [int(row['category_it'] == i) for i in VERONA_DATASET_CATEGORIES],
        axis = 1
    )
    node_category_attr_dict = dict(zip(poi_data['id'], poi_data['tourism_category']))

    #%%
    POI_graph, updated_distance_matrix = distance_matrix_to_poi_graph(
        distance_matrix,
        threshold = 5000,
        node_min_visit_time_attr_dict = node_min_visit_time_attr_dict,
        node_max_visit_time_attr_dict = node_max_visit_time_attr_dict
    )
    poi_data['_osm_id'] = poi_data['id'].apply(lambda x: poiid2idx[x])


    #%%

    # insert weather in checkins
    weather_rain_dict = weather_df.to_dict()['rain']
    weather_temp_dict = weather_df.to_dict()['temp']

    df_poi_train_firstpart = pd.read_csv(
        os.path.join(DATASET_PATH, 'data_train_first_part.csv')
    ).drop('Unnamed: 0', axis = 1)
    df_poi_train_firstpart = df_poi_train_firstpart.drop_duplicates(inplace = False)
    print(f'df_poi_train_firstpart.shape {df_poi_train_firstpart.shape}')


    df_poi_train_secondpart = pd.read_csv(
        os.path.join(DATASET_PATH, 'data_train_second_part.csv')
    ).drop('Unnamed: 0', axis = 1).drop_duplicates(inplace = False)
    print(f'df_poi_train_secondpart.shape {df_poi_train_secondpart.shape}')

    df_poi_train_raw = pd.concat(
        [df_poi_train_firstpart, df_poi_train_secondpart]
    ).sort_values(
        ['id_veronacard', 'data_visita', 'ora_visita']
    ).drop_duplicates(inplace = False)
    df_poi_train_raw['rain'] = df_poi_train_raw['data_visita'].apply(
        lambda d: weather_rain_dict.get(d, None)
    )
    df_poi_train_raw['temp'] = df_poi_train_raw['data_visita'].apply(
        lambda d: weather_temp_dict.get(d, None)
    )
    print(f'df_poi_train_raw.shape {df_poi_train_raw.shape}')

    #%%
    df_poi_train_raw['date'] = pd.to_datetime(df_poi_train_raw['data_visita'])

    #%%
    df_poi_train_raw_2018 = df_poi_train_raw[
        (df_poi_train_raw['date'].dt.year == 2018) &
        (df_poi_train_raw['date'].dt.month >= 5) &
        (df_poi_train_raw['date'].dt.month <=8) &
        (df_poi_train_raw['rain'] == 'no_rain')
    ]
    df_poi_train_raw_2019 = df_poi_train_raw[
        (df_poi_train_raw['date'].dt.year == 2019) &
        (df_poi_train_raw['date'].dt.month >= 5) &
        (df_poi_train_raw['date'].dt.month <=8) &
        (df_poi_train_raw['rain'] == 'no_rain')
    ]
    #%%
    print(f'df_poi_train_raw_2018.shape {df_poi_train_raw_2018.shape}')
    print(f'df_poi_train_raw_2019.shape {df_poi_train_raw_2019.shape}')


    #%%
    ##
    # Create train and test set user requests with start loc, end loc and time budget
    ##
    df_poi_test = verona_df_to_request_df(
        df_poi_train_raw_2019,
        poi_data,
        distance_matrix,
        poiid2idx,
        MIN_TC_REQUESTS = 20,
        RANDOM_SEED = SEED
    )
    print(f'df_poi_test.shape {df_poi_test.shape}')
    test_samples = df_poi_test.shape[0]

    #%%
    df_poi_train_val = verona_df_to_request_df(
        df_poi_train_raw_2018,
        poi_data,
        distance_matrix,
        poiid2idx,
        MIN_TC_REQUESTS = 85, # since they will be further split into train and val set, we need to ensure that we have enough samples
        RANDOM_SEED = SEED,
        MAX_TRAIN_REQUESTS = None
    )
    df_poi_train = df_poi_train_val.sample(frac = 0.875, random_state = SEED)
    df_poi_val = df_poi_train_val.drop(df_poi_train.index)
    print(f'df_poi_train.shape {df_poi_train.shape}')
    print(f'df_poi_val.shape {df_poi_val.shape}')
    
    #%%
    # train unique POIs
    unique_pois = np.sort(df_poi_train['route_pois'].explode().unique())
    print(f'Unique POIs {unique_pois}')

    #%%
    # Drop POIs from data which are not present in train-test set
    poi_data = poi_data[poi_data['id'].isin(unique_pois)].copy(deep = True)

    #%%
    df_poi_train['route_pois_w_ids'] = df_poi_train['route_pois'].apply(
        lambda x: [poiid2idx[p] for p in x]
    )
    df_poi_test['route_pois_w_ids'] = df_poi_test['route_pois'].apply(
        lambda x: [poiid2idx[p] for p in x]
    )


    # POI-POI transition matrix
    train_poi_transition_count_matrix, train_poi_transition_prob_matrix = poi_transition_matrix(
        df_poi_train,
        'route_pois_w_ids',
        14
    )
    # POI-POI transition matrix
    test_poi_transition_count_matrix, test_poi_transition_prob_matrix = poi_transition_matrix(
        df_poi_test,
        'route_pois_w_ids',
        14
    )

    #%%
    ##
    # poi weights based on number of departures from a POI
    ##
    poi_departure_counts = [
        sum(r) for r in test_poi_transition_count_matrix
    ]
    poi_weights = poi_departure_counts / sum(poi_departure_counts)

    jensen_shannon_distances = []
    jensen_shannon_div = []
    weighted_jensen_shannon_distances = []
    weighted_jensen_shannon_div = []
    for row_idx in range(train_poi_transition_count_matrix.shape[0]):

        dist = distance.jensenshannon(
            train_poi_transition_prob_matrix[row_idx],
            test_poi_transition_prob_matrix[row_idx],
            2
        )
        jensen_shannon_distances.append(dist)
        jensen_shannon_div.append(dist**2)

        weighted_jensen_shannon_distances.append(dist * poi_weights[row_idx])
        weighted_jensen_shannon_div.append((dist**2) * poi_weights[row_idx])

    print(jensen_shannon_distances)
    print(f'Average jensen_shannon_distances {sum(jensen_shannon_distances)/(len(jensen_shannon_distances))}')
    print(f'Average jensen_shannon_divergence {sum(jensen_shannon_div)/(len(jensen_shannon_div))}')
 
        
    print(f'Average weighted jensen_shannon_distances {sum(weighted_jensen_shannon_distances)/(len(weighted_jensen_shannon_distances))}')
    print(f'Average weighted jensen_shannon_divergence {sum(weighted_jensen_shannon_div)/(len(weighted_jensen_shannon_div))}')

    #%%
    # visualise heat maps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(train_poi_transition_prob_matrix, ax=axes[0], cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Train POI-POI transition matrix")
    sns.heatmap(test_poi_transition_prob_matrix, ax=axes[1], cmap="viridis", vmin=0, vmax=1,
                cbar_kws={"shrink": 0.8})
    axes[1].set_title("Test data POI-POI transition matrix")
    plt.tight_layout()
    plt.show()

    #%%
    # Compute POI popularity based on test set
    n_routes = df_poi_test.shape[0]
    counts = Counter()
    for r in df_poi_test['route_pois']:
        counts.update(set(r))
    
    poi_data['popularity'] = poi_data['id'].apply(
        lambda poiid: counts[poiid]/ n_routes
    )
    
    poi_data['importance_score'] = poi_data['popularity']
    #%%
    
    # Category transition matrix
    poiid2cat = {}
    for idx, row in poi_data.iterrows():
        poiid2cat[int(row.id)] = row.category_it

    df_poi_test['routes_as_cats'] = df_poi_test['route_pois'].apply(
        lambda r: [VERONA_DATASET_CATEGORIES2IDS[poiid2cat[p]] for p in r]
    )
    category_count_matrix, category_transition_matrix = poi_transition_matrix(
        df_poi_test,
        'routes_as_cats',
        3
    )


    #%%
    # duplicate test set
    test_set_w_duplicate_requests = pd.DataFrame(
        columns = ['req_id', 'start_node', 'end_node', 'time_constraint', 'route_pois']
    )

    req_idx = 0
    for row in df_poi_test.itertuples():
        start_node = row.start_node
        end_node = row.end_node
        tc = row.time_constraint
        route_pois = row.route_pois
        
        for i in range(duplicate_copies):
            test_set_w_duplicate_requests.loc[req_idx] = [
                req_idx, start_node, end_node, tc, route_pois
            ]
            req_idx += 1
    print(f'test_set_w_duplicate_requests.shape {test_set_w_duplicate_requests.shape}')


    #%%
    # save all files
    ##
    # POI trans prob matrix, Category trans. prob. matrix
    ##
    poi_data.to_csv(os.path.join(SAVE_PATH, 'final_pois.csv'))
    # ox.io.save_graphml(walking_network, os.path.join(SAVE_PATH, 'walking_network.graphml'))
    with open(os.path.join(SAVE_PATH,'start_node_pois_within_radius.pkl'), 'wb') as f:
        pickle.dump(start_node_poi_ids, f)
    with open(os.path.join(SAVE_PATH,'train_start_node_pois.pkl'), 'wb') as f:
        pickle.dump(start_node_poi_ids, f)

    indexing_dicts = {'idx2poiid': idx2poiid, 'poiid2idx': poiid2idx}
    with open(os.path.join(SAVE_PATH,'indexing_dicts.pkl'), 'wb') as f:
        pickle.dump(indexing_dicts, f)
    with open(os.path.join(SAVE_PATH,'distance_matrix.npy'), 'wb') as f:
        np.save(f, distance_matrix)
    with open(os.path.join(SAVE_PATH, 'node_category_attr_dict.pkl'), 'wb') as f:
        pickle.dump(node_category_attr_dict, f)

    with open(os.path.join(SAVE_PATH,'bearing_matrix.npy'), 'wb') as f:
        np.save(f, bearing_matrix)
    with open(os.path.join(SAVE_PATH,'train_poi_transition_count_matrix.npy'), 'wb') as f:
        np.save(f, train_poi_transition_count_matrix)
    with open(os.path.join(SAVE_PATH,'train_poi_transition_prob_matrix.npy'), 'wb') as f:
        np.save(f, train_poi_transition_prob_matrix)
    with open(os.path.join(SAVE_PATH,'test_poi_transition_count_matrix.npy'), 'wb') as f:
        np.save(f, test_poi_transition_count_matrix)
    with open(os.path.join(SAVE_PATH,'test_poi_transition_prob_matrix.npy'), 'wb') as f:
        np.save(f, test_poi_transition_prob_matrix)
    with open(os.path.join(SAVE_PATH,'category_count_matrix.npy'), 'wb') as f:
        np.save(f, category_count_matrix)
    with open(os.path.join(SAVE_PATH,'category_transition_matrix.npy'), 'wb') as f:
        np.save(f, category_transition_matrix)
    
    # save distance matrix, POI graph and node category dict for different thresholds
    for POI_graph_threshold in [500, 1000, 3000, 5000]:

        POI_graph, updated_distance_matrix = distance_matrix_to_poi_graph(
            distance_matrix,
            threshold = POI_graph_threshold,
            node_min_visit_time_attr_dict = node_min_visit_time_attr_dict,
            node_max_visit_time_attr_dict = node_max_visit_time_attr_dict
        )
        
        os.makedirs(os.path.join(SAVE_PATH, f'poi_graph_{POI_graph_threshold}'), exist_ok = True)
        with open(os.path.join(SAVE_PATH, f'poi_graph_{POI_graph_threshold}', 'node_category_attr_dict.pkl'), 'wb') as f:
            pickle.dump(node_category_attr_dict, f)
        nx.write_graphml(POI_graph, os.path.join(SAVE_PATH, f'poi_graph_{POI_graph_threshold}', 'POI_graph_updated.graphml'))
        with open(os.path.join(SAVE_PATH, f'poi_graph_{POI_graph_threshold}', 'distance_matrix_updated.npy'), 'wb') as f:
            np.save(f, updated_distance_matrix)

    #%%
    df_poi_train.to_csv(os.path.join(SAVE_PATH, 'train_set.csv'))
    df_poi_val.to_csv(os.path.join(SAVE_PATH, 'val_set.csv'))
    df_poi_test.to_csv(os.path.join(SAVE_PATH, 'test_set_w_single_reqs.csv'))
    test_set_w_duplicate_requests.to_csv(os.path.join(SAVE_PATH, 'test_set.csv'))
    

# %%
