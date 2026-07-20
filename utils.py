import pandas as pd
import numpy as np

import osmnx as ox
import networkx as nx
from shapely.geometry import Point
import numpy as np
import pickle

import matplotlib.pyplot as plt
import geopandas as gpd
import pygeohash as pgh

import os
import copy

import optuna

from constants import FINAL_CATEGORIES, NEW_VISIT_DURATION_BASED_ON_CATEGORIES

SEED = 100
np.random.seed(SEED)
GEOHASH_PRECISION = 7

def read_data(IMPL_DIRECTORY, POI_graph_threshold = None):
    '''
    Read data files
    '''
    print('read_data()')

    final_pois = pd.read_csv(os.path.join(IMPL_DIRECTORY,'saved_data/final_pois.csv'))
    print(f'final_pois.shape {final_pois.shape}')
    final_pois = preprocess_final_pois(final_pois)

    walking_network = None

    with open(os.path.join(IMPL_DIRECTORY,'saved_data/start_node_pois_within_radius.pkl'), 'rb') as f:
        start_node_pois_within_radius = pickle.load(f)
    print(f'len(start_node_pois_within_radius) {len(start_node_pois_within_radius)}')

    with open(os.path.join(IMPL_DIRECTORY,'saved_data/indexing_dicts.pkl'), 'rb') as f:
        indexing_dicts = pickle.load(f)
    print(f'indexing_dicts {len(indexing_dicts)}')
    print(f"idx2poiid {len(indexing_dicts['idx2poiid'])}")
    print(f"poiid2idx {len(indexing_dicts['poiid2idx'])}")
    idx2poiid = indexing_dicts['idx2poiid']
    poiid2idx = indexing_dicts['poiid2idx']

    with open(os.path.join(IMPL_DIRECTORY,'saved_data/distance_matrix.npy'), 'rb') as f:
        unfiltered_distance_matrix = np.load(f)
    print(f'unfiltered_distance_matrix {unfiltered_distance_matrix.shape}')

    if POI_graph_threshold is not None:
        with open(os.path.join(IMPL_DIRECTORY, 'saved_data', f'poi_graph_{POI_graph_threshold}', 'node_category_attr_dict.pkl'), 'rb') as f:
            node_category_attr_dict = pickle.load(f)
        print(f'len(node_category_attr_dict) {len(node_category_attr_dict)}')

        POI_graph = nx.read_graphml(os.path.join(IMPL_DIRECTORY, 'saved_data', f'poi_graph_{POI_graph_threshold}', 'POI_graph_updated.graphml'), node_type = int)
        print(f'POI_graph {POI_graph}')

        with open(os.path.join(IMPL_DIRECTORY,'saved_data', f'poi_graph_{POI_graph_threshold}', 'distance_matrix_updated.npy'), 'rb') as f:
            distance_matrix = np.load(f)
        print(f'distance_matrix {distance_matrix.shape}')
    else:
        with open(os.path.join(IMPL_DIRECTORY, 'saved_data', 'node_category_attr_dict.pkl'), 'rb') as f:
            node_category_attr_dict = pickle.load(f)
        print(f'len(node_category_attr_dict) {len(node_category_attr_dict)}')

        POI_graph = nx.read_graphml(os.path.join(IMPL_DIRECTORY, 'saved_data', 'POI_graph_updated.graphml'), node_type = int)
        print(f'POI_graph {POI_graph}')

        with open(os.path.join(IMPL_DIRECTORY,'saved_data', 'distance_matrix_updated.npy'), 'rb') as f:
            distance_matrix = np.load(f)
        print(f'distance_matrix {distance_matrix.shape}')   

    with open(os.path.join(IMPL_DIRECTORY, 'saved_data/bearing_matrix.npy'), 'rb') as f:
        bearing_matrix = np.load(f)
    print(f'bearing_matrix {bearing_matrix.shape}')



    return (
        final_pois, walking_network, start_node_pois_within_radius,
        idx2poiid, poiid2idx, unfiltered_distance_matrix, node_category_attr_dict,
        POI_graph, distance_matrix, bearing_matrix
    )
    


def preprocess_final_pois(final_pois):

    print('preprocess_final_pois()')

    final_pois['tourism_category'] = final_pois['tourism_category'].apply(
        lambda x: [int(i) for i in x.strip('[].').split('. ')]
    )

    final_pois["geometry"] = gpd.GeoSeries.from_wkt(final_pois["geometry"])
    final_pois_gdf = gpd.GeoDataFrame(final_pois, geometry = "geometry")

    # (lat, lon)
    final_pois_gdf['plotting_coords'] = final_pois_gdf['geometry'].apply(lambda x: (x.y, x.x) if isinstance(x, Point) else (x.centroid.y, x.centroid.x))
    
    # compute geohashes
    final_pois_gdf['geohash'] = final_pois_gdf['plotting_coords'].apply(
        lambda x: pgh.encode(x[0], x[1], precision = GEOHASH_PRECISION)
    )

    print(f'final_pois_gdf.shape {final_pois_gdf.shape}')

    return final_pois_gdf



def rescale_inputs(
        POI_graph,
        final_pois,
        poiid2idx,
        distance_matrix,
        unfiltered_distance_matrix,
        start_node_pois_within_radius,
        train_set_start_node_osmids = []
    ):

    print('rescale_inputs()')

    ##
    # Create new POI graph with ids from 1
    ##
    _POI_graph = nx.relabel_nodes(POI_graph, poiid2idx)
    print(f'{_POI_graph}')
    print(f'_POI_graph.nodes[0] {_POI_graph.nodes[0]}')

    # rescale min_visit_time and max_visit_time to hours
    for node in _POI_graph.nodes:
        _POI_graph.nodes[node]['min_visit_time'] /= 3600
        _POI_graph.nodes[node]['max_visit_time'] /= 3600
    print(f'_POI_graph.nodes[0], _POI_graph.nodes[50] {_POI_graph.nodes[0], _POI_graph.nodes[50]}')

    # rescale distance matrix to km
    distance_matrix = distance_matrix/1000
    unfiltered_distance_matrix = unfiltered_distance_matrix/1000

    _start_node_pois_within_radius = [poiid2idx[s] for s in start_node_pois_within_radius]
    print(f'len(_start_node_pois_within_radius) {len(_start_node_pois_within_radius)}')

    _train_set_start_node_osmids = None
    if len(train_set_start_node_osmids) > 0:
        _train_set_start_node_osmids = [poiid2idx[s] for s in train_set_start_node_osmids]
        print(f'len(_train_set_start_node_osmids) {len(_train_set_start_node_osmids)}')

    return (
        _POI_graph,
        final_pois,
        distance_matrix,
        unfiltered_distance_matrix,
        _start_node_pois_within_radius,
        _train_set_start_node_osmids
    )

def normalise_inputs(distance_matrix, unfiltered_distance_matrix = None):

    # ignoring np.inf for normalization
    if distance_matrix is not None:
        matrix_with_nan = np.where(np.isinf(distance_matrix), np.nan, distance_matrix)
        _distance_matrix = (matrix_with_nan - np.nanmin(matrix_with_nan)) / (np.nanmax(matrix_with_nan) - np.nanmin(matrix_with_nan))

    return _distance_matrix

##
# Evaluation related utilities
##
def check_positions(vectors, second_vector):
    # Combine the vectors by performing a logical OR on each position
    combined_vector = np.max(vectors, axis=0)
    
    for i in range(len(second_vector)):
        if second_vector[i] == 1 and combined_vector[i] != 1:
            return False
    return True


def distance_matrix_to_poi_graph(distance_matrix, poi_data, idx2poiid, threshold = 7500):
    '''
    Create POI graph from distance matrix
    '''
    updated_distance_matrix = copy.deepcopy(distance_matrix)
    updated_distance_matrix[updated_distance_matrix > threshold] = np.inf

    POI_graph = nx.Graph()
    # insert edges in POI_graph if distance is non-zero and not np.inf in distance_matrix
    for i, source in idx2poiid.items():
        POI_graph.add_node(source)
        for j, target in idx2poiid.items():
            if i != j and updated_distance_matrix[i, j] != np.inf:
                POI_graph.add_edge(source, target, weight = updated_distance_matrix[i, j])
    
    print(f'POI_graph.number_of_nodes(), POI_graph.number_of_edges() {POI_graph.number_of_nodes(), POI_graph.number_of_edges()}')
    print(f'Min and max degree {min([x[1] for x in list(POI_graph.degree())]), max([x[1] for x in list(POI_graph.degree())])}')
    print(f'Max Min edge weight {max(data['weight'] for u, v, data in POI_graph.edges(data=True)), min(data['weight'] for u, v, data in POI_graph.edges(data=True))}')

    node_min_visit_time_attr_dict, node_max_visit_time_attr_dict, node_category_attr_dict = get_visit_time_dicts(POI_graph, poi_data)

    nx.set_node_attributes(POI_graph, node_min_visit_time_attr_dict, 'min_visit_time')
    nx.set_node_attributes(POI_graph, node_max_visit_time_attr_dict, 'max_visit_time')

    return POI_graph, updated_distance_matrix, node_category_attr_dict


def get_visit_time_dicts(POI_graph, poi_data):

    node_category_attr_dict = dict.fromkeys(list(dict(POI_graph.nodes(data=True)).keys()))
    
    for node_id in node_category_attr_dict.keys():
        node_category_attr_dict[node_id] = list(poi_data[poi_data['osm_id'] == node_id]['tourism_category'].iloc[0])
    
    node_min_visit_time_attr_dict = dict.fromkeys(list(dict(POI_graph.nodes()).keys()))
    node_max_visit_time_attr_dict = dict.fromkeys(list(dict(POI_graph.nodes()).keys()))
    for node_id in node_min_visit_time_attr_dict.keys():

        category_vector = node_category_attr_dict[node_id]
        category_indices_from_vector = list((np.array(category_vector) == 1).nonzero()[0])

        min_visit_time = min([
            NEW_VISIT_DURATION_BASED_ON_CATEGORIES[FINAL_CATEGORIES[idx]][0] for idx in category_indices_from_vector
        ])

        max_visit_time = max([
            NEW_VISIT_DURATION_BASED_ON_CATEGORIES[FINAL_CATEGORIES[idx]][1] for idx in category_indices_from_vector
        ])
        node_min_visit_time_attr_dict[node_id] = min_visit_time
        node_max_visit_time_attr_dict[node_id] = max_visit_time

    return node_min_visit_time_attr_dict, node_max_visit_time_attr_dict, node_category_attr_dict


def delete_study_if_exists(study_name: str, storage_url: str):
    try:
        # Get all studies from the storage
        studies = optuna.study.get_all_study_summaries(storage=storage_url)

        # Check if the study exists
        if any(study.study_name == study_name for study in studies):
            print(f"Study '{study_name}' exists. Deleting...")
            optuna.delete_study(study_name=study_name, storage=storage_url)
            print("Deleted successfully.")
        else:
            print(f"Study '{study_name}' does not exist. Nothing to delete.")
    except Exception as e:
        print(f"Error: {e}")




    

    





