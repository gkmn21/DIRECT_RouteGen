import math
import numpy as np
from numba import njit
from scipy.special import softmax
from scipy.spatial import distance as scp_dist
import pandas as pd
import gymnasium as gym
import folium
import copy

import os
import time
import json
import pygeohash as pgh

from constants import FINAL_CATEGORIES, VERONA_DATASET_CATEGORIES


SEED = 100
np.random.seed(SEED)
INCLUDE_KEYS = [
    'element',
    'id',
    'addr:city',
    'addr:country',
    'amenity',
    'tourism'
]

#########################
# Arc computation
EARTH_R = 6371000.0  # meters
def arc_latlon_points(
    s,
    e,
    arc_length,
    n = 100,
    side = None,
    degenerate_bearing_deg = None
):
    """
    s, e: (lat, lon) in degrees
    arc_length: desired arc length in meters
    n: number of output points
    side: +1 or -1, controls arc direction
    degenerate_bearing_deg:
        Used only when s == e.
        Direction in degrees from s to the circle center.
    """

    lat1, lon1 = map(math.radians, s)
    lat2, lon2 = map(math.radians, e)

    lat0 = 0.5 * (lat1 + lat2)
    lon0 = 0.5 * (lon1 + lon2)

    sx = EARTH_R * (lon1 - lon0) * math.cos(lat0)
    sy = EARTH_R * (lat1 - lat0)
    ex = EARTH_R * (lon2 - lon0) * math.cos(lat0)
    ey = EARTH_R * (lat2 - lat0)

    vx = ex - sx
    vy = ey - sy
    d = math.hypot(vx, vy)

    # Degenerate case: s == e
    if d < 1e-9:
        if arc_length <= 0:
            raise ValueError("arc_length must be positive when s == e")

        r = arc_length / (2 * math.pi)

        bearing = math.radians(degenerate_bearing_deg)

        # Bearing convention: 0° = north, 90° = east
        cx = sx + r * math.sin(bearing)
        cy = sy + r * math.cos(bearing)

        # Angle from circle center to start point s
        start_angle = math.atan2(sy - cy, sx - cx)

        points = []

        for i in range(n):
            t = i / (n - 1) if n > 1 else 0

            # Full circular arc starting and ending at s
            a = start_angle + 2 * math.pi * t

            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)

            lat = y / EARTH_R + lat0
            lon = x / (EARTH_R * math.cos(lat0)) + lon0

            points.append((math.degrees(lat), math.degrees(lon)))

        # Force exact start/end coordinates
        points[0] = s
        points[-1] = e

        return points

    if arc_length <= d:
        raise ValueError("arc_length must be greater than straight-line distance")

    target = arc_length / d
    lo = 1e-12
    hi = 2 * math.pi - 1e-12

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        value = mid / (2 * math.sin(mid / 2))

        if value < target:
            lo = mid
        else:
            hi = mid

    theta = 0.5 * (lo + hi)
    r = arc_length / theta

    mx = 0.5 * (sx + ex)
    my = 0.5 * (sy + ey)

    nx = -vy / d
    ny = vx / d

    h = math.sqrt(max(0.0, r * r - (d / 2) ** 2))

    cx = mx + side * h * nx
    cy = my + side * h * ny

    a1 = math.atan2(sy - cy, sx - cx)
    a2 = math.atan2(ey - cy, ex - cx)

    ccw_delta = (a2 - a1) % (2 * math.pi)

    if abs(ccw_delta - theta) <= abs((2 * math.pi - ccw_delta) - theta):
        direction = 1
    else:
        direction = -1

    points = []

    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        a = a1 + direction * theta * t

        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)

        lat = y / EARTH_R + lat0
        lon = x / (EARTH_R * math.cos(lat0)) + lon0

        points.append((math.degrees(lat), math.degrees(lon)))

    return points

def circle_selection(start_node, end_node, circle_params, idx_to_coords, geohash_to_idx):

    circles_info_dict = dict.fromkeys(
        (
            [-1, 1] if start_node != end_node else [0, 90, 180, 270]
        )    
    )

    circle_pois_count = []
    for key in circles_info_dict.keys():

        if start_node != end_node:
            circle_points = arc_latlon_points(
                idx_to_coords[start_node],
                idx_to_coords[end_node],
                arc_length = circle_params['arc_length'],
                n = circle_params['n_cuts'] + 2, # for start and end points
                side = key,
                degenerate_bearing_deg = None
            )
        else:
            circle_points = arc_latlon_points(
                idx_to_coords[start_node],
                idx_to_coords[end_node],
                arc_length = circle_params['arc_length'],
                n = circle_params['n_cuts'] + 2, # for start and end points
                side = None,
                degenerate_bearing_deg = key
            )

        # create dictionary of unique geohashes for circle points
        circle_geohashes = dict.fromkeys([
            pgh.encode(
                coords[0], coords[1], precision = circle_params['geohash_precision']
            ) for coords in circle_points
        ])

        # compute POI count in each geohash cell of the circle
        for geohash in circle_geohashes.keys():
            if geohash[:circle_params['geohash_precision']] in geohash_to_idx:
                circle_geohashes[geohash] = len(geohash_to_idx[geohash[:circle_params['geohash_precision']]])
            else:
                circle_geohashes[geohash] = 0
        
        circle_pois_count.append(sum(circle_geohashes.values()))
        circles_info_dict[key] = (
            circle_points,
            circle_geohashes
        )
    
    # sample circle based on poi count as probabilities
    try:
        total = sum([c for c in circle_pois_count if c > 2]) # only consider circles with more than 2 POIs for sampling
        if total == 0:
            total = sum(circle_pois_count)
            if total == 0:
                selected_circle_key = np.random.choice(list(circles_info_dict.keys()))
            else:
                selected_circle_key = np.random.choice(
                    list(circles_info_dict.keys()),
                    p = [c / total for c in circle_pois_count]
                )
        else:
            selected_circle_key = np.random.choice(
                list(circles_info_dict.keys()),
                p = [c / total if c > 2 else 0 for c in circle_pois_count]
            )
    except Exception as e:
        print(f"Error occurred while selecting circle: {e}")
        print(f"Circle POI counts: {circle_pois_count}")
    # select circle with highest POI counts
    # selected_circle_key = list(circles_info_dict.keys())[np.argmax(circle_pois_count)]

    return circles_info_dict[selected_circle_key]

#########################
@njit
def score_candidates(distances, diversity_deltas, coverage_deltas, cat_pref_scores, alpha_diversity, alpha_distance, alpha_coverage, alpha_cat_pref):
    
    out = np.empty_like(distances)

    for i in range(len(distances)):
        out[i] = (
            (alpha_diversity * diversity_deltas[i]) + (alpha_coverage * coverage_deltas[i]) + (alpha_cat_pref * cat_pref_scores[i])
        )
    return out

@njit
def calculate_turn_angle_based_penalty(bearing1, bearing2):
    turn_angle = abs(bearing2 - bearing1)
    penalty = turn_angle/360
        
    return -1 * penalty

@njit
def normalise_value(value, max_value):
    # avoid division by zero
    if max_value == 0:
        return -1
    return value / max_value
###


def total_intra_list_distance(vectors):
    packed = [int(''.join('1' if x else '0' for x in v), 2) for v in vectors]
    total = 0
    n = len(packed)
    for i in range(n):
        for j in range(i+1, n):
            total += (packed[i] ^ packed[j]).bit_count()
    return total

def compute_ild(vectors):
    '''
    Normalised ILD
    '''

    n = len(vectors)
    tot = total_intra_list_distance(vectors)
    return tot / (n*(n-1)*len(vectors[0])/2)

###

class Route:

    def __init__(
        self,
        start_node,
        end_node,
        walking_speed = 5 #5km/h
    ):
        
        self.walking_speed = walking_speed
        self.start_node = start_node
        self.end_node = end_node
        self.route = [self.start_node] # Route is a list of nodes
        self.time_elapsed = 0
        self.distance_elapsed = 0


    def __repr__(self):
        print(f"Start Node: {self.start_node}")
        print(f"End Node: {self.end_node}")
        print(f"Route: {self.route}")
        print(f"Time Elapsed: {self.time_elapsed}")
        print(f"Distance Elapsed: {self.distance_elapsed}")

        return str(self.route)


    def insert_node(self, node, distance_of_new_node, visit_time_of_new_node):
        '''
        Insert a node in route and update time_elapsed, distance_elapsed
        '''

        self.distance_elapsed += distance_of_new_node
        self.time_elapsed += (distance_of_new_node/self.walking_speed) + visit_time_of_new_node

        self.route.append(node)



    def remove_node(self, distance_of_removed_node, visit_time_of_removed_node):
        removed_node = self.route.pop(-1)
        self.distance_elapsed -= distance_of_removed_node
        self.time_elapsed -= (distance_of_removed_node/self.walking_speed) + visit_time_of_removed_node
        return removed_node


class CityEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        city_graph = None,
        all_start_nodes = None,
        bearing_matrix = None,
        poiid2idx = None,
        final_pois_gdf = None,
        unfiltered_distance_matrix = None,
        output_dir = None,
        train_samples = None,
        current_mode = None,
        max_city_graph_nodes = None,
        candidate_poi_generator_k = 3,
        alpha_params_dict = {
            'temporal_distance': 1,
            'diversity': 0.33,
            'coverage': 0.33, 
            'cat_prefs': 0.33
        },
        end_node_variant = False,
        circle_params = {
            'density_weight': 1,
            'n_cuts': 10,
            'arc_length': 3000,
            'geohash_precision': 6
        },
        render_neighbors = True,
        render_mode = None
    ):
        self.walking_speed = 5 #5km/h
        self.all_start_nodes = all_start_nodes # list of all start nodes for masking
        self.current_mode = current_mode
        self.train_iter = None
        self.train_samples = train_samples
        self.original_graph = city_graph
        self.city_graph = copy.deepcopy(city_graph) # original graph with start nodes
        # remove all start nodes from city graph, and then insert each start node in reset() 
        # and remove the start node on termination
        if not end_node_variant:
            self.city_graph.remove_nodes_from(self.all_start_nodes)
        self.max_city_graph_nodes = max_city_graph_nodes
        self.nodes_count = self.city_graph.number_of_nodes()
        self.edges_count = self.city_graph.number_of_edges()
        self.request_id = None
        self.render_neighbors = render_neighbors

        self.end_node_variant = end_node_variant
        self.circle_params = circle_params
        self.circle_step = None
        self.circle_points = None
        self.circle_geohashes = None
        
        
        self.poi_limit	= 50 # poi_limit # limit for observation space
        self.bearing_matrix = bearing_matrix
        self.unfiltered_distance_matrix = unfiltered_distance_matrix
        self.poiid2idx = poiid2idx
        self.final_pois_gdf = final_pois_gdf
        if self.end_node_variant:
            self.idx_to_tourism_category = {
                row['_osm_id'] : np.array(row['tourism_category'], dtype = np.float32)
                for _, row in self.final_pois_gdf.iterrows()
            }
        else:
            self.idx_to_tourism_category = {
                row['_osm_id'] : np.array(row['tourism_category'][4:], dtype = np.float32)
                for _, row in self.final_pois_gdf.iterrows()
            }
        
        # dirty check for verona dataset based on end_node_variant
        if self.end_node_variant:
            self.tourism_category_arr = np.zeros((len(self.poiid2idx), len(VERONA_DATASET_CATEGORIES)), dtype=np.float32)
        else:
            self.tourism_category_arr = np.zeros((len(self.poiid2idx), len(FINAL_CATEGORIES[4:])), dtype=np.float32)
        
        for poiid, idx in self.poiid2idx.items():
            self.tourism_category_arr[idx] = self.idx_to_tourism_category[idx]

        self.sorted_neighbors_dict = {}
        for poiid in self.original_graph.nodes():
            idx = self.poiid2idx[poiid]
            row = self.unfiltered_distance_matrix[idx]
            nbrs = list(self.original_graph.neighbors(poiid))
            self.sorted_neighbors_dict[idx] = sorted(
                ((nbr, row[self.poiid2idx[nbr]]) for nbr in nbrs),
                key=lambda x: x[1]
            )
        self.current_graph_nodes = None
        self.avg_transit_time = None
        self.avg_visit_time = None

        self.route_instance = None
        self.distance_from_end_node = None
        self.constraints_dict = None

        self.terminated = False
        self.reward = 0
        self.output_dir = output_dir

        self.map = None

        self.candidate_poi_generator_k = candidate_poi_generator_k
        self.alpha_params_dict = alpha_params_dict
        self.sparse_cell_nodes = dict()
        self.previously_inserted_pois_tracker = set()
        self.selected_neighbors = None
        self.neighbor_nodes_scores = None
        self.all_neighbors_sorted_by_distances = None
        self.route_diversity = None
        self.route_poi_geohashes_list = None
        self.idx_to_geohash = {
            row['_osm_id'] : row['geohash']
            for _, row in self.final_pois_gdf.iterrows()
        }
        # index POIs for each geohash, at circle_param precision
        self.geohash_to_idx = {}
        precision = self.circle_params['geohash_precision']
        for idx, geohash in self.idx_to_geohash.items():
            if geohash[:precision] not in self.geohash_to_idx:
                self.geohash_to_idx[geohash[:precision]] = []

            if self.end_node_variant:
                self.geohash_to_idx[geohash[:precision]].append(idx)
            else:
                if idx not in self.all_start_nodes:
                    self.geohash_to_idx[geohash[:precision]].append(idx)
        self.idx_to_coords = {
            row['_osm_id'] : row['plotting_coords']
            for _, row in self.final_pois_gdf.iterrows()
        }
        self.circle_budget = None
        self.selected_neighbors_dist_benefit = None
        self.selected_neighbors_time_benefit = None
        self.selected_neighbors_turn_angle = None

        self.invalid_action_flag = False

        self.episode_step_counter = None

        # insert, delete, go to end location
        self.action_space = gym.spaces.Discrete(self.candidate_poi_generator_k + 2)

        self.observation_space = gym.spaces.Dict(
            {
                'route_nodes': gym.spaces.Box(low=min(self.city_graph.nodes), high=self.max_city_graph_nodes+1, shape=(self.poi_limit,), dtype= np.int16), #high=max(self.city_graph.nodes)+1
                'distance_elapsed': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # max distance of route is 15 km (10km for 2hrs)
                'time_elapsed': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # max duration of route is 10 hrs
                'distance_from_end_node': gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'temporal_distance_from_end_node': gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'poi_count': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
                'selected_neighbors_dist_benefit': gym.spaces.Box(low=-1, high=1, shape=(self.candidate_poi_generator_k,), dtype= np.float32),
                'selected_neighbors_time_benefit': gym.spaces.Box(low=-1, high=1, shape=(self.candidate_poi_generator_k,), dtype= np.float32),
                'selected_neighbors_turn_angle': gym.spaces.Box(low =-1, high=0, shape=(self.candidate_poi_generator_k,), dtype=np.float32)
            }
        )

    
    def _get_obs(self):
        '''
        Convert environment's state into observation and
        return the current observation
        '''
        
        # pad route for route_nodes
        route = self.route_instance.route
        if len(route) < self.poi_limit:
            PAD_NODE_ID = self.max_city_graph_nodes
            route = route + [PAD_NODE_ID for _ in range(self.poi_limit - len(route))]

        # pad neighbor node scores
        neighbor_nodes_scores = self.neighbor_nodes_scores
        if len(neighbor_nodes_scores) < self.candidate_poi_generator_k:
            neighbor_nodes_scores = np.append(neighbor_nodes_scores, [0 for _ in range(self.candidate_poi_generator_k - len(self.neighbor_nodes_scores))]).astype(np.float32)
        
        # pad neighbors dist and time benefit
        selected_neighbors_dist_benefit = self.selected_neighbors_dist_benefit
        if len(selected_neighbors_dist_benefit) < self.candidate_poi_generator_k:
            selected_neighbors_dist_benefit = np.append(selected_neighbors_dist_benefit, [-1 for _ in range(self.candidate_poi_generator_k - len(self.selected_neighbors_dist_benefit))]).astype(np.float32)
        selected_neighbors_time_benefit = self.selected_neighbors_time_benefit
        if len(selected_neighbors_time_benefit) < self.candidate_poi_generator_k:
            selected_neighbors_time_benefit = np.append(selected_neighbors_time_benefit, [-1 for _ in range(self.candidate_poi_generator_k - len(self.selected_neighbors_time_benefit))]).astype(np.float32)
        selected_neighbors_turn_angle = self.selected_neighbors_turn_angle
        if len(selected_neighbors_turn_angle) < self.candidate_poi_generator_k:
            selected_neighbors_turn_angle = np.append(selected_neighbors_turn_angle, [-1 for _ in range(self.candidate_poi_generator_k - len(self.selected_neighbors_turn_angle))]).astype(np.float32)
        
        # normalise distance_elapsed and distance_from_end_node
        _distance_elapsed = np.clip(normalise_value(self.route_instance.distance_elapsed, self.constraints_dict['distance_constraint']),
        0, 1
        )

        _time_elapsed = np.clip(normalise_value(self.route_instance.time_elapsed, self.constraints_dict['time_constraint']),
        0, 1
        )
        remaining_distance_budget = self.constraints_dict['distance_constraint'] - self.route_instance.distance_elapsed
        remaining_time_budget = self.constraints_dict['time_constraint'] - self.route_instance.time_elapsed
        _distance_from_end_node = np.clip(
            normalise_value(remaining_distance_budget - self.distance_from_end_node, remaining_distance_budget),
            -1, 1
        )
        _temporal_distance_from_end_node = np.clip(
            normalise_value(remaining_time_budget - (self.distance_from_end_node / self.walking_speed), remaining_time_budget),
            -1, 1
        )


        # normalise poi count
        _poi_count = normalise_value(len(self.route_instance.route), self.poi_limit)

        obs = {
            'route_nodes': np.array([np.array(x) for x in route], dtype = np.int16),
            'distance_elapsed': np.array([_distance_elapsed], dtype=np.float32),
            'time_elapsed': np.array([_time_elapsed], dtype=np.float32),
            'distance_from_end_node': np.array([_distance_from_end_node], dtype=np.float32),
            'temporal_distance_from_end_node': np.array([_temporal_distance_from_end_node], dtype=np.float32),
            'poi_count': np.array([_poi_count], dtype=np.int8),
            'selected_neighbors_dist_benefit': selected_neighbors_dist_benefit,
            'selected_neighbors_time_benefit': selected_neighbors_time_benefit,
            'selected_neighbors_turn_angle': selected_neighbors_turn_angle
        }
        

        return obs



    def reset(self, seed = None, options = {'test_sample_parameters': None}):
        '''
        Resets the environment
        '''

        super().reset(seed = seed)
        if seed:
            np.random.seed(seed)

        self.previously_inserted_pois_tracker = set()
        self.episode_step_counter = 0

        # initialise route instance
        start_node = 1 # None
        if self.end_node_variant:
            end_node = 5 # None
        time_constraint = 2 # None
        cat_prefs = None
        cat_prefs_binary = None
        if options:
            if options.get('test_sample_parameters', None):
                start_node = options['test_sample_parameters']['start_node']
                if self.end_node_variant:
                    end_node = options['test_sample_parameters']['end_node']
                time_constraint = options['test_sample_parameters']['time_constraint']
                self.request_id = options['test_sample_parameters']['request_id']
                cat_prefs = options['test_sample_parameters']['cat_prefs']

        elif self.current_mode == 'train':
            self.train_iter = (self.train_iter + 1)%len(self.train_samples) if self.train_iter is not None else 0
            start_node = self.train_samples.iloc[self.train_iter]['start_node_ids']
            if self.end_node_variant:
                end_node = self.train_samples.iloc[self.train_iter]['end_node_ids']
            time_constraint = self.train_samples.iloc[self.train_iter]['time_constraint']
            cat_prefs = self.train_samples.iloc[self.train_iter]['cat_prefs']
       

        cat_prefs_binary = None
        # dirty check for verona dataset
        if self.end_node_variant == False:   
            end_node = start_node
            cat_prefs_binary = None if cat_prefs is None else np.array([1 if i in cat_prefs else 0 for i in FINAL_CATEGORIES[4:]], dtype = np.float32) 

        
        # insert start node and its attributes and edges in city graph
        if not self.end_node_variant:
            self.city_graph.add_node(start_node, **self.original_graph.nodes[start_node])
            self.city_graph.add_edges_from((start_node, nbr, self.original_graph[start_node][nbr]) for nbr in self.original_graph.neighbors(start_node) if nbr not in self.all_start_nodes)
        self.current_graph_nodes = set(self.city_graph.nodes())

        # walkable distance
        distance_constraint = self.walking_speed * time_constraint if time_constraint <= 2 else 15

        # set avg_visit time and transit time
        # computing average visiting time (in hours) and transit time (hours) for a city
        self.avg_visit_time = sum(data['min_visit_time'] for _, data in self.city_graph.nodes(data = True)) / self.city_graph.number_of_nodes()
        self.avg_transit_time = (sum(
            data['weight'] for _, _, data in self.city_graph.edges(data=True)
        ) / self.city_graph.number_of_edges())/ (1000 * self.walking_speed)

        self.constraints_dict = {
            'time_constraint': time_constraint,
            'distance_constraint': distance_constraint,
            'cat_prefs': cat_prefs_binary
        }
        
        self.route_instance = Route(
            start_node,
            end_node,
            self.walking_speed
        )
        # self.distance_from_end_node = 0
        self.distance_from_end_node = self.unfiltered_distance_matrix[self.poiid2idx[self.route_instance.route[-1]]][self.poiid2idx[self.route_instance.end_node]]
        #self.rejected_nodes = set([])
        self.route_poi_geohashes_list = [self.idx_to_geohash[start_node]]

        # compute search circle, excluding start and end points
        self.sparse_cell_nodes = dict()

        ##
        # Probabilistic circle selection based on POI counts
        ##
        self.circle_points, self.circle_geohashes = circle_selection(
            start_node, end_node, self.circle_params, self.idx_to_coords, self.geohash_to_idx
        ) 

        # budget for search in circle
        self.circle_budget = self.constraints_dict['time_constraint']

        # for each cell, save cell_visit_time proportion
        # convert counts to proportions and save t_max in same circle_geohashes
        total_pois = sum(self.circle_geohashes.values())
        density_weight = self.circle_params['density_weight']
        uniform_moving_proportion = 1 / len(self.circle_geohashes)
        if total_pois > 0:
            t_max = 0
            cumulative_proportion = 0 
            for geohash in self.circle_geohashes:
                # cell density
                cell_density = self.circle_geohashes[geohash] / total_pois

                # cell_visit_proportion
                cell_visit_proportion = (density_weight * cell_density) + ((1 - density_weight) * uniform_moving_proportion)
                cumulative_proportion += cell_visit_proportion
                t_max = (cumulative_proportion * self.circle_budget)

                self.circle_geohashes[geohash] = t_max

        # index of current geohash in self.circle_geohashes
        self.circle_step = 0


        self.route_diversity = self.compute_route_ild(self.route_instance.route)
        (
            self.selected_neighbors,
            self.neighbor_nodes_scores,
            self.all_neighbors_sorted_by_distances,
            self.selected_neighbors_dist_benefit,
            self.selected_neighbors_time_benefit,
            self.selected_neighbors_turn_angle
        )= self.candidate_poi_generator(self.candidate_poi_generator_k)

        observation = self._get_obs()


        if self.current_mode != 'train':
            # initialise map with all pois for visualisation
            #(lat, lon)
            start_node_coordinates = self.final_pois_gdf[self.final_pois_gdf['_osm_id'] == self.route_instance.start_node]['plotting_coords'].values[0]
            self.map = folium.Map(
                location=[start_node_coordinates[0], start_node_coordinates[1]], zoom_start=20
            )

        self.terminated = False
        self.reward = 0

        info = {
            'start_node': start_node,
            'time_constraint': time_constraint,
            'distance_constraint': distance_constraint,
            'cat_prefs': cat_prefs_binary,
            'route': self.route_instance.route,
            'action_return_tuple': None
        }
        return observation, info
        
    
    def candidate_poi_generator(self, k):
        '''
        Generates POI candidates for insert operation
        '''

        request_cat_prefs = self.constraints_dict.get('cat_prefs', None)

        current_node = self.route_instance.route[-1]
        start_node = self.route_instance.start_node
        end_node = self.route_instance.end_node

        current_node_neighbors = self.sorted_neighbors_dict[current_node]
        remaining_distance_budget = self.constraints_dict['distance_constraint'] - self.route_instance.distance_elapsed
        remaining_time_budget = self.constraints_dict['time_constraint'] - self.route_instance.time_elapsed
        

        # Filter: remove previously traversed nodes and self.rejected_nodes from neighbors
        route_set = set(self.route_instance.route)
        route_geohash_set = set(self.route_poi_geohashes_list)
        all_neighbors_sorted_by_distances = [
           (nbr, dist)
           for nbr, dist in current_node_neighbors
           if (
               nbr in self.current_graph_nodes and nbr not in route_set and nbr != end_node
            )
        ]

        # consider POIs in current cell, 
        # if less than k neighbors in current cell consider next cell, and so on, until k neighbors are found or all cells are exhausted
        # current_cell_visit_proportion = self.circle_geohashes[current_geohash]
        circle_geohash_precision = self.circle_params['geohash_precision']
        # if no neighbors are present in current cell, go to next step cell
        if len(all_neighbors_sorted_by_distances) > 0:
            
            while True:
                current_geohash = list(self.circle_geohashes.keys())[self.circle_step]
                __all_neighbors_sorted_by_distances = [
                    (nbr, dist)
                    for nbr, dist in all_neighbors_sorted_by_distances
                    if self.idx_to_geohash[nbr][:circle_geohash_precision] == current_geohash[:circle_geohash_precision]
                ] + [(k, v) for k, v in self.sparse_cell_nodes.items()]
                __all_neighbors_sorted_by_distances = sorted(__all_neighbors_sorted_by_distances, key=lambda x: x[1])
                
                if len(__all_neighbors_sorted_by_distances) == 0 and self.circle_step < (len(self.circle_geohashes) - 1):
                    self.circle_step = self.circle_step + 1
                
                elif len(__all_neighbors_sorted_by_distances) < k:
                    # put current neighbors in list and continue to next cell
                    self.sparse_cell_nodes.update(__all_neighbors_sorted_by_distances)
                    # if we haven't found enough neighbors, continue to the next cell while considering current cell neighbors
                    if self.circle_step < (len(self.circle_geohashes) - 1):
                        self.circle_step = self.circle_step + 1
                    else:
                        # if we have exhausted all cells, break
                        all_neighbors_sorted_by_distances = __all_neighbors_sorted_by_distances
                        break
                else:
                    # neighbors found in current cell, break
                    all_neighbors_sorted_by_distances = __all_neighbors_sorted_by_distances
                    break

        current_node_neighbors = [
            nbr for nbr, _ in all_neighbors_sorted_by_distances
        ]


        if len(current_node_neighbors) == 0:
            return [], np.array([], dtype=np.float32), all_neighbors_sorted_by_distances, np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)



        # Candidate Scoring
        current_node_neighbors_scores = []
        current_node_neighbors_idx_dict = dict.fromkeys(current_node_neighbors, None)
        distances = []
        diversity_deltas = []
        coverage_deltas = []
        cat_pref_scores = []

        # benefit of visiting neighbor and returning to start
        neighbors_distance_benefit = []
        neighbors_time_benefit = []
        neighbors_turn_angle = []
        route_with_two_or_more_nodes = len(self.route_instance.route) > 1
        if route_with_two_or_more_nodes:
            poi1 = self.poiid2idx[self.route_instance.route[-2]]
            poi2 = self.poiid2idx[self.route_instance.route[-1]]

        for idx, neighbor in enumerate(current_node_neighbors):

            ##
            # Category diversity score
            ##
            diversity_after_neighbor_insertion = self.compute_route_ild(self.route_instance.route + [neighbor])
            diversity_deltas.append(
                0.5 * (1 + (diversity_after_neighbor_insertion - self.route_diversity))
            )


            ##
            # Coverage score
            ##
            coverage_deltas.append(1 - int(self.idx_to_geohash[neighbor] in route_geohash_set))

            ##
            # Temporal Distance score (walking distance + visiting time)
            ##
            neighbor_visit_time = self.city_graph.nodes[neighbor].get('min_visit_time', 0)
            neighbor_distance = self.unfiltered_distance_matrix[self.poiid2idx[current_node]][self.poiid2idx[neighbor]]
            distances.append((neighbor_distance/self.walking_speed) + neighbor_visit_time)
            
            ##
            # Benefit of visiting neighbor (for returning in observation space)
            ##
            neighbor_distance_cost = (
                self.unfiltered_distance_matrix[self.poiid2idx[current_node]][self.poiid2idx[neighbor]]
                +
                self.unfiltered_distance_matrix[self.poiid2idx[neighbor]][self.poiid2idx[end_node]] 
            )
            neighbor_distance_benefit = np.clip(
                normalise_value(remaining_distance_budget - neighbor_distance_cost, remaining_distance_budget),
                -1, 1
            )
            neighbors_distance_benefit.append(neighbor_distance_benefit)
            neighbor_time_cost = (neighbor_distance_cost/self.walking_speed) + neighbor_visit_time
            neighbor_time_benefit = np.clip(
                normalise_value(remaining_time_budget - neighbor_time_cost, remaining_time_budget),
                -1, 1
            )
            neighbors_time_benefit.append(neighbor_time_benefit)

            # turn angle benefit
            if route_with_two_or_more_nodes:
                poi3 = self.poiid2idx[neighbor]
                neighbors_turn_angle.append(calculate_turn_angle_based_penalty(
                    self.bearing_matrix[poi1][poi2],
                    self.bearing_matrix[poi2][poi3]
                ))
            else:
                neighbors_turn_angle.append(0)

            ##
            # Category Prefs
            ##
            cat_pref_score = 0
            if not (request_cat_prefs is None) and sum(request_cat_prefs) > 0 and sum(self.tourism_category_arr[neighbor]) > 0:
                #cat_pref_score = 1 if np.any(np.logical_and(self.tourism_category_arr[neighbor], request_cat_prefs)) else 0
                cat_pref_score = 1 - scp_dist.dice(self.tourism_category_arr[neighbor], request_cat_prefs)
            cat_pref_scores.append(cat_pref_score)


            current_node_neighbors_idx_dict[neighbor] = idx 

        current_node_neighbors_scores = score_candidates(
            np.array(distances, dtype = np.float64),
            np.array(diversity_deltas, dtype = np.float64),
            np.array(coverage_deltas),
            np.array(cat_pref_scores, dtype = np.float64),
            np.float32(self.alpha_params_dict['diversity']),
            np.float32(self.alpha_params_dict['temporal_distance']),
            np.float32(self.alpha_params_dict['coverage']),
            np.float32(self.alpha_params_dict['cat_prefs'])
        )

        # Neighbor scores to probabilities
        if np.isnan(cat_pref_scores).any():
            print(cat_pref_scores)
            print(request_cat_prefs)
        current_node_neighbors_probs = softmax(current_node_neighbors_scores)
        
        # Sample k neighbors based on probabilities
        non_zero_probs = len(current_node_neighbors_probs[current_node_neighbors_probs > 0])
        selected_neighbors = np.random.choice(
            current_node_neighbors,
            size = k if non_zero_probs >= k else non_zero_probs,
            replace = False,
            p = current_node_neighbors_probs
        )
        selected_neighbor_probs = np.array(
            [current_node_neighbors_probs[current_node_neighbors_idx_dict[s]] for s in selected_neighbors],
            dtype = np.float32
        )

        selected_neighbors_dist_benefit = np.array(
            [neighbors_distance_benefit[current_node_neighbors_idx_dict[s]] for s in selected_neighbors],
            dtype = np.float32
        )
        selected_neighbors_time_benefit = np.array(
            [neighbors_time_benefit[current_node_neighbors_idx_dict[s]] for s in selected_neighbors],
            dtype = np.float32
        )
        selected_neighbors_turn_angle = np.array(
            [neighbors_turn_angle[current_node_neighbors_idx_dict[s]] for s in selected_neighbors],
            dtype = np.float32
        )


        return selected_neighbors, selected_neighbor_probs, all_neighbors_sorted_by_distances, selected_neighbors_dist_benefit, selected_neighbors_time_benefit, selected_neighbors_turn_angle


    
    def compute_route_ild(self, route):
        '''
        Remove start node and compute ild of route
        '''
        ild = 0
        if len(route) > 2:
            _route = route[1:]
            cats = self.tourism_category_arr[_route] 
            ild = compute_ild(cats)
        return ild




    def perform_action(self, action):

        action_return_tuple = None
        current_node = self.route_instance.route[-1]


        # Action value in [0, k-1] and node has neighbors: Insert node in route
        # Note: Any one of the top k neighbors can be inserted
        if action in range(self.candidate_poi_generator_k) and action < len(self.selected_neighbors):
            self.invalid_action_flag = False
            
            inserted_node = self.selected_neighbors[action]
            distance_of_inserted_node = None
            for neigh, dist in self.all_neighbors_sorted_by_distances:
                if neigh == inserted_node:
                    distance_of_inserted_node = dist
                    break
            self.route_instance.insert_node(
                inserted_node, # inserted node
                distance_of_inserted_node, # distance of inserted node from current node
                self.city_graph.nodes[inserted_node].get('min_visit_time', 0) # visit duration
            )
            self.distance_from_end_node = self.unfiltered_distance_matrix[self.poiid2idx[self.route_instance.route[-1]]][self.poiid2idx[self.route_instance.end_node]]


            # update route_poi_geohashes_list, selected neighbors, scores, route diversity
            self.route_poi_geohashes_list.append(self.idx_to_geohash[inserted_node])
            inserted_node_score = self.neighbor_nodes_scores[action]
            # self.route_diversity = self.compute_alpha_ndcg(self.route_instance.route)
            self.route_diversity = self.compute_route_ild(self.route_instance.route)
            
            # If inserted node is in sparse_cell_nodes, remove all nodes appearing before inserted node and inserted node
            keys_list = list(self.sparse_cell_nodes.keys())
            if inserted_node in keys_list:
                for key in keys_list[:keys_list.index(inserted_node) + 1]:
                    del self.sparse_cell_nodes[key]
            else:
                self.sparse_cell_nodes.clear()

            # increase circle_step after insertion, based on proportion
            if self.route_instance.time_elapsed >= self.circle_geohashes[list(self.circle_geohashes.keys())[self.circle_step]]:
                while True:
                    if self.circle_step < (len(self.circle_geohashes) - 1):
                        self.circle_step = self.circle_step + 1
                        if self.route_instance.time_elapsed < self.circle_geohashes[list(self.circle_geohashes.keys())[self.circle_step]]:
                            break
                    else:
                        break
                    
            

            # generate POI candidates
            (
                self.selected_neighbors,
                self.neighbor_nodes_scores,
                self.all_neighbors_sorted_by_distances,
                self.selected_neighbors_dist_benefit,
                self.selected_neighbors_time_benefit,
                self.selected_neighbors_turn_angle 
            )= self.candidate_poi_generator(self.candidate_poi_generator_k)

            action_return_tuple = ('insert', inserted_node, inserted_node_score)


        # Remove last node from route if route length > 1
        elif action == self.candidate_poi_generator_k and (len(self.route_instance.route) > 1):
            self.invalid_action_flag = False

            # print('ACTION: REMOVE LAST NODE')
            removed_node = self.route_instance.remove_node(
                self.unfiltered_distance_matrix[self.poiid2idx[self.route_instance.route[-2]]][self.poiid2idx[self.route_instance.route[-1]]],
                self.city_graph.nodes[self.route_instance.route[-1]].get('min_visit_time', 0)
            )

            # decrease circle_step if necessary after deletion
            while True:
                if self.circle_step != 0 and self.route_instance.time_elapsed < self.circle_geohashes[list(self.circle_geohashes.keys())[self.circle_step - 1]]:
                    self.circle_step = self.circle_step - 1
                else:
                    break

            self.distance_from_end_node = self.unfiltered_distance_matrix[self.poiid2idx[self.route_instance.route[-1]]][self.poiid2idx[self.route_instance.end_node]]
            self.route_poi_geohashes_list.pop(-1)

            # update neighbor node scores
            self.route_diversity = self.compute_route_ild(self.route_instance.route)
            (
                self.selected_neighbors,
                self.neighbor_nodes_scores,
                self.all_neighbors_sorted_by_distances,
                self.selected_neighbors_dist_benefit,
                self.selected_neighbors_time_benefit,
                self.selected_neighbors_turn_angle
            )= self.candidate_poi_generator(self.candidate_poi_generator_k)
            action_return_tuple = ('remove', removed_node)

        # Action value 2: Go to end node
        elif action == (self.candidate_poi_generator_k + 1) and (len(self.route_instance.route) > 1):
            self.invalid_action_flag = False
            action_return_tuple = ('end_node')
            distance_from_current_node_to_end_node = self.unfiltered_distance_matrix[self.poiid2idx[current_node]][self.poiid2idx[self.route_instance.end_node]]
            self.route_instance.insert_node(
                self.route_instance.end_node,
                distance_from_current_node_to_end_node,
                0
            )
            self.distance_from_end_node = 0
        else:
            self.invalid_action_flag = True

        return action_return_tuple
    
    

    def get_reward(self, action_return_tuple):

        if self.invalid_action_flag:
            return -100

        reward = 0
        reward_components_counter = 0

        distance_constraint = self.constraints_dict['distance_constraint']
        time_constraint = self.constraints_dict['time_constraint']

        # smooth function for penalty based on time
        margin = time_constraint - (self.route_instance.time_elapsed + (2 * self.avg_transit_time) + self.avg_visit_time)
        score = 1 / (1 + np.exp(-margin))

        if not self.terminated:

            ##
            # Component A: time margin penalty
            ##
            if score <= 0.5:
                reward -= (1 - score)


            ##
            # Component B: Reward for insertion
            # Rewards/penalties on insertion
            ##
            if action_return_tuple and action_return_tuple[0] == 'insert':
                # +1 point for each inserted node
                inserted_node_score = 1
                if action_return_tuple[1] not in self.previously_inserted_pois_tracker:
                    self.previously_inserted_pois_tracker.add(action_return_tuple[1])
                    reward += inserted_node_score
                
                # based on bearing
                if len(self.route_instance.route) > 2:
                    poi1 = self.poiid2idx[self.route_instance.route[-3]]
                    poi2 = self.poiid2idx[self.route_instance.route[-2]]
                    poi3 = self.poiid2idx[self.route_instance.route[-1]]
                    reward += calculate_turn_angle_based_penalty(
                        self.bearing_matrix[poi1][poi2],
                        self.bearing_matrix[poi2][poi3]
                    )
                    reward_components_counter += 1

            ##
            # Checking internally for truncation
            ##
            elif self.episode_step_counter >= 50:
                reward -= 100


        ##
        # On Termination
        ##
        else:
            # Component C: Reward for reaching end node
            reward_components_counter += 1
            if self.route_instance.end_node == self.route_instance.route[-1] and len(self.route_instance.route) > 1 and self.route_instance.distance_elapsed <= distance_constraint and self.route_instance.time_elapsed <= time_constraint:
                reward += (100 * (len(self.route_instance.route) - 2))
            else:
                reward -= 100

        self.reward = reward
        return self.reward
    

    def step(self, action):
        '''
        Perform action and compute state of environment
        '''

        action_return_tuple = self.perform_action(action)
        self.episode_step_counter += 1

        observation = self._get_obs()

        # Episode terminates when any of the below conditions are True:
        # 1. agent violates time or distance constraints
        # 2. agent reaches end_node
        terminated = False
        distance_constraint = self.constraints_dict['distance_constraint']
        time_constraint = self.constraints_dict['time_constraint']
        if (
            self.route_instance.distance_elapsed >= distance_constraint
        ) or (
            self.route_instance.time_elapsed >= time_constraint
        ) or (
            self.route_instance.end_node == self.route_instance.route[-1] and len(self.route_instance.route) > 1
        ):
            terminated = True
            self.terminated = True
            if not self.end_node_variant:
                self.city_graph.remove_node(self.route_instance.start_node)
            self.current_graph_nodes = []

        reward = self.get_reward(action_return_tuple)
        truncated = False

        # returning intermediate route in info dictionary, incase episode gets truncated
        info = {
            'route': self.route_instance.route,
            'action_return_tuple': action_return_tuple
        }
        if self.terminated:
            info = {
                'start_node': self.route_instance.start_node,
                'end_node': self.route_instance.end_node,
                'time_constraint': self.constraints_dict['time_constraint'],
                'distance_constraint': self.constraints_dict['distance_constraint'],
                'route': self.route_instance.route,
                'action_return_tuple': action_return_tuple
            }

        return observation, reward, terminated, truncated, info
    


    def render(self, mode = 'human'):
        '''
        Render the environment
        '''

        identifier = self.request_id

        if mode == 'rgb_array':
            return np.zeros((64, 64, 3), dtype=np.uint8)

        elif self.current_mode != 'train': 

            # compute route segment distances
            route_segment_distances = []
            start_node_idx = self.route_instance.route[0]
            end_node_idx = self.route_instance.route[-1]
            first_poi_idx = self.route_instance.route[1] if len(self.route_instance.route) > 1 else None
            route_distance = self.unfiltered_distance_matrix[self.poiid2idx[start_node_idx]][self.poiid2idx[first_poi_idx]] if len(self.route_instance.route) > 1 else 0
            route_segment_distances.append(route_distance)

            if len(self.route_instance.route) > 2:
                for idx in range(1, len(self.route_instance.route) - 1):
                    segment_distance = self.unfiltered_distance_matrix[
                        self.route_instance.route[idx]][ self.route_instance.route[idx + 1]
                    ]
                    route_distance += segment_distance
                    route_segment_distances.append(segment_distance)


            # render pois
            if self.render_neighbors:
                for idx in self.city_graph.nodes():
                    row = self.final_pois_gdf[self.final_pois_gdf['_osm_id'] == idx].iloc[0]
                    osm_id = row.get('_osm_id', None)
                    popularity = row.get('importance_score', None)

                    if osm_id in self.route_instance.route:
                        continue
                    poi_visit_duration = self.city_graph.nodes[osm_id].get('min_visit_time', 0) if (
                        osm_id in self.current_graph_nodes
                    ) else 0

                    reduced_keys = []
                    for key in row.keys():
                        if key not in INCLUDE_KEYS:
                            continue

                        if isinstance(row[key], list) and not pd.isna(row[key]).all():
                            reduced_keys.append(key)
                        elif not pd.isna(row[key]):
                            reduced_keys.append(key)

                    tooltip_string = ''.join([
                        '<p><strong>'+str(key) + '</strong>'+ ':' + str(row[key]).translate(str.maketrans({'`': '', '´': ''})) + '</p>' for key in reduced_keys
                    ])
                    tooltip_string = tooltip_string + '<p><strong>Visit duration: '+ str(poi_visit_duration) + '</strong></p>''<p><strong>Popularity: '+ str(popularity) + '</strong></p>'

                    folium.Marker(
                        location = [row['plotting_coords'][0],row['plotting_coords'][1]],
                        icon = folium.Icon(color = 'red', icon = 'circle-dot', angle = 0, prefix = 'fa'),
                        tooltip = tooltip_string
                    ).add_to(self.map)

            # render route
            polyline_coords = []
            for route_poi_idx, osm_id in enumerate(self.route_instance.route):
                row = self.final_pois_gdf[self.final_pois_gdf['_osm_id'] == osm_id].iloc[0]
                visit_time = self.city_graph.nodes[osm_id].get('min_visit_time', 0) if route_poi_idx != 0 and route_poi_idx != (len(self.route_instance.route) - 1)  else 0

                reduced_keys = []
                for key in row.keys():
                    if key not in INCLUDE_KEYS:
                        continue

                    if isinstance(row[key], list) and not pd.isna(row[key]).all():
                        reduced_keys.append(key)
                    elif not pd.isna(row[key]):
                        reduced_keys.append(key)


                # remove characters which cause issues with html rendering
                tooltip_string = f'<p><strong>Route POI {route_poi_idx}</strong></p><p><strong>POI ID {osm_id}</strong></p><p>Visit Time {visit_time}'+'</p>'+''.join([
                    '<p><strong>'+str(key) + '</strong>'+ ':' + str(row[key]).translate(str.maketrans({'`': '', '´': ''})) + '</p>' for key in reduced_keys
                ])

                # Route start and end nodes in different color
                if route_poi_idx == 0 or route_poi_idx == len(self.route_instance.route) - 1:
                    location_type = 'Start' if route_poi_idx == 0 else 'End'
                    tooltip_string = f'<p><strong>{location_type}</strong></p>' + tooltip_string
                    folium.Marker(
                        location = [row['plotting_coords'][0],row['plotting_coords'][1]],
                        icon = folium.Icon(color = 'green', icon = 'house', angle = 0, prefix = 'fa'),
                        tooltip = tooltip_string
                    ).add_to(self.map)

                else:
                    folium.Marker(
                        location = [row['plotting_coords'][0],row['plotting_coords'][1]],
                        icon = folium.Icon(color = 'blue', icon = 'circle-dot', angle = 0, prefix = 'fa'),
                        tooltip = tooltip_string
                    ).add_to(self.map)

                polyline_coords.append([row['plotting_coords'][0], row['plotting_coords'][1]])

            for i in range(len(polyline_coords) - 1): 
                polyline_coord1 = polyline_coords[i]
                polyline_coord2 = polyline_coords[i + 1]
                tooltip_text_distance = route_segment_distances[i]  
                tooltip_text_duration = route_segment_distances[i] / self.walking_speed
                folium.PolyLine(
                    [polyline_coord1, polyline_coord2],
                    color = "blue",
                    weight = 2.5,
                    opacity = 1,
                    tooltip = f'{round(tooltip_text_distance, 2)} km, {round(tooltip_text_duration, 2)} h'
                ).add_to(self.map)

            div_string = f'<div style="position: absolute; top: 10px; left: 10px; background-color: rgba(255, 255, 255, 0.7); padding: 10px; font-size: 16px; font-weight: bold; border-radius: 5px; z-index: 9999;">ID: {identifier}<br>Start: {self.route_instance.start_node}<br>End: {self.route_instance.end_node}<br>Time Elapsed: {self.route_instance.time_elapsed} <br> Time Constraint: {self.constraints_dict["time_constraint"]}<br>Route Distance: {self.route_instance.distance_elapsed}<br> Distance Constraint: {self.constraints_dict["distance_constraint"]} <br> Last Reward: {self.reward} </div>'

            label = folium.Element(div_string)
            popup = folium.Popup(label, max_width=300)

            ##
            # Add circle arc
            ##
            # Draw arc
            # folium.PolyLine(
            #     locations = self.circle_points,
            #     color = "red",
            #     weight = 4,
            #     opacity = 0.8,
            #     tooltip = "Arc"
            # ).add_to(self.map)
            # # Add small markers for generated arc points
            # for i, p in enumerate(self.circle_points):
            #     folium.CircleMarker(
            #         location=p,
            #         radius=2,
            #         color="red",
            #         fill=True,
            #         fill_opacity=0.7,
            #         popup=f"Point {i}"
            #     ).add_to(self.map)
            

            # Add the label to the map (use it as a custom overlay)
            self.map.get_root().html.add_child(folium.Element(div_string))

            output_file = f'{time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(time.time()))}.html'
            if self.output_dir:
                if identifier is not None:
                    output_file = os.path.join(self.output_dir, f'{identifier}_{output_file}')
                else:
                    output_file = os.path.join(self.output_dir, output_file)
            self.map.save(output_file)

        
        return str(self.route_instance.route)
    





                
