from pprint import pprint, pformat
from typing import Any, SupportsFloat

import numpy as np
from numpy.random import randint
from scipy.spatial.distance import cdist, euclidean

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import gymnasium
from gymnasium import spaces


class Delivery(gymnasium.Env):
    def __init__(self, n_stops=10, max_demand=10, max_vehicle_cap=30, max_env_size=1_000_000,
                 gen_seed=None, gym_seed=None) -> None:
        # Data generation
        self.n_stops = n_stops
        self.max_demand = max_demand
        self.max_vehicle_cap = max_vehicle_cap
        self.max_env_size = max_env_size
        self.gen_seed = gen_seed
        self.gym_seed = gym_seed
        assert self.max_demand <= self.max_vehicle_cap
        
        self.ortools_data, self.other_data = self._create_data()

        # Gymnasium env
        self.observation, self.info = self.reset()

    
    def _generate_demands_and_vehicle_caps(self):
        demands = np.append([0], randint(1, self.max_demand + 1, size=self.n_stops - 1))

        vehicle_caps = np.full(shape=self.n_stops, fill_value=randint(self.max_demand, self.max_vehicle_cap))

        return demands, vehicle_caps
    
        
    def _create_data(self):
        ortools_data = dict()
        other_data = dict()
        
        np.random.seed(self.gen_seed)
        
        stops_coords = randint(0, self.max_env_size + 1, size=(self.n_stops, 2))
        demands, vehicle_caps = self._generate_demands_and_vehicle_caps()

        other_data['stops_coords'] = stops_coords

        ortools_data['distance_matrix'] = cdist(stops_coords, stops_coords)
        ortools_data['demands'] = demands
        ortools_data['vehicle_caps'] = vehicle_caps
        ortools_data['num_vehicles'] = len(vehicle_caps)
        ortools_data['depot'] = 0

        return ortools_data, other_data
        

    def visualize(self, show=True, save_path=None):
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

        ax1.scatter(self.other_data['stops_coords'][0, 0], self.other_data['stops_coords'][0, 1], color='red', s=10)
        ax1.scatter(self.other_data['stops_coords'][1:, 0], self.other_data['stops_coords'][1:, 1], color='black', s=8)
        for i, txt in enumerate(self.ortools_data['demands']):
            ax1.annotate(txt, (self.other_data['stops_coords'][i, 0], self.other_data['stops_coords'][i, 1]), fontsize=8, textcoords="offset points", xytext=(0, 2), ha='center')
        min_size = np.min(self.other_data['stops_coords'])
        max_size = np.max(self.other_data['stops_coords'])
        ax1.set_xlim([min_size, max_size])
        ax1.set_ylim([min_size, max_size])

        if show:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path, dpi=300)

        return fig, ax1


    def _solve_ortools(self, time_limit_seconds=10):
        # Instantiate the data problem.
        print(f'Solving with time limit of {time_limit_seconds} seconds')

        data = self.ortools_data
        data['distance_matrix'] = data['distance_matrix'].astype('int')
        data['demands'] = data['demands']
        data['vehicle_caps'] = data['vehicle_caps']

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_caps"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        return data, manager, routing, solution


    def get_ortools_solution(self, time_limit_seconds=10):
        data, manager, routing, solution = self._solve_ortools(time_limit_seconds=time_limit_seconds)
        
        total_distance = 0
        total_load = 0  
        
        routes = list()

        for vehicle_id in range(data["num_vehicles"]):
            route = dict()
            
            route_distance = 0
            route_load = 0
            route_segments = list()

            index = routing.Start(vehicle_id)            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)

                route_load += data["demands"][node_index]                
                route_segments.append(dict(index=node_index, load=route_load))

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            route_segments.append(dict(index=manager.IndexToNode(index), load=route_load))

            route['segments'] = route_segments
            route['distance'] = route_distance
            routes.append(route)

            total_distance += route_distance
            total_load += route_load

        return dict(
            objective=solution.ObjectiveValue(),
            total_distance=total_distance,
            total_load=total_load,
            routes=routes
        )


    def visualize_ortools_solution(self, solution=None, time_limit_seconds=10, show=True):
        _color_list = list(mcolors.TABLEAU_COLORS.keys())
        fig, ax1 = self.visualize(show=False)
        if solution is None:
            solution = self.get_ortools_solution(time_limit_seconds=time_limit_seconds)
        routes = filter(lambda route: route['distance'] > 0, solution['routes'])

        for _i, route in enumerate(routes):
            _color = _color_list[_i % len(_color_list)]
            route_segments = route['segments']
            route_distance = route['distance']
            route_load = route_segments[-1]['load']
            route_coords = self.other_data['stops_coords'][[x['index'] for x in route_segments], :]

            ax1.plot(route_coords[:, 0], route_coords[:, 1], linewidth=1, color=_color)
            ax1.annotate(f'{route_load}',
                         ((route_coords[-1, 0] + route_coords[-2, 0])/2, (route_coords[-1, 1] + route_coords[-2, 1])/2),
                         fontsize=8, textcoords="offset points", xytext=(0, 0), ha='center', color=_color)

        if show:
            plt.show()


    def reset(self):
        super().reset(seed=self.gym_seed)

        info = dict()

        self.action_space = spaces.Discrete(self.n_stops, seed=self.gym_seed)        
        self.observation_space = spaces.Dict({
            'coord': spaces.Box(
                low=0, high=self.max_env_size,
                shape=(self.n_stops, 2), dtype=np.int32,
                seed=self.gym_seed),
            'demand': spaces.Box(
                low=0, high=self.max_demand,
                shape=(self.n_stops, ), dtype=np.int32,
                seed=self.gym_seed),
            'visited': spaces.MultiBinary(self.n_stops, seed=self.gym_seed),
                # visited of stop 0 should always be 0
            'current_load': spaces.Box(
                low=0, high=self.max_vehicle_cap,
                shape=(1, ), dtype=np.int32,
                seed=self.gym_seed),
            'current_stop': spaces.Discrete(self.n_stops, seed=self.gym_seed),
            'current_length': spaces.Box(
                low=0, high=np.inf,
                shape=(1, ), dtype=np.int32,
                seed=self.gym_seed)
        })
        self.reward_range = (-np.inf, 0)

        observation = dict(
            coord=self.other_data['stops_coords'].astype('int32'),
            demand=self.ortools_data['demands'].astype('int32'),
            current_load=self.ortools_data['vehicle_caps'][0],
            visited=np.zeros(self.n_stops).astype('int32'),
            current_stop=0,
            current_length=0
        )
        
        return observation, info
    

    def step(self, action):
        observation = self.observation
        info = self.info
        # print('action', action)
        # pprint(observation)

        assert self.action_space.contains(action), f"Invalid action {action}"
        assert observation['visited'][action] == 0, f"Stop {action} has already been visited"

         
        # move to next stop
        if action != 0:
            observation['visited'][action] = 1
        observation['current_length'] += int(euclidean(observation['coord'][observation['current_stop']], observation['coord'][action]))
        observation['current_load'] -= observation['demand'][action]
        assert observation['current_load'] >= 0, f"Current load cannot be negative"
        observation['current_stop'] = action


        # reset load if back to depot
        if observation['current_stop'] == 0:
            observation['current_load'] = self.ortools_data['vehicle_caps'][0]

        # check if done
        if np.sum(observation['visited']) == self.n_stops - 1 and observation['current_stop'] == 0:
            reward = -observation['current_length']
            terminated = True
        else:
            reward = 0
            terminated = False
        
        return (
            observation,
            reward,
            terminated,
            False, # truncated (bool) – Whether the truncation condition outside the scope of the MDP is satisfied.
            info
        )
    

if __name__ == "__main__":
    # delivery = Delivery(n_stops=10)
    # pprint(delivery.other_data)
    # pprint(delivery.ortools_data)
    # delivery.visualize()
    # print(delivery.observation)
    # delivery.step(1)
    # delivery.step(2)
    # pprint(delivery.observation)
    # pprint(delivery.get_ortools_solution(time_limit_seconds=1))
    # pprint(delivery.visualize_ortools_solution(time_limit_seconds=1))

    delivery = Delivery(n_stops=15, gen_seed=42, gym_seed=42)
    delivery.visualize()
    pprint(solution := delivery.get_ortools_solution(time_limit_seconds=10))
    delivery.visualize_ortools_solution(solution=solution)
    
    delivery.step(2)
    delivery.step(11)
    delivery.step(13)
    delivery.step(0)

    delivery.step(8)
    delivery.step(9)
    delivery.step(10)
    delivery.step(0)

    delivery.step(7)
    delivery.step(0)

    delivery.step(5)
    delivery.step(6)
    delivery.step(14)
    delivery.step(12)
    delivery.step(0)

    delivery.step(4)
    delivery.step(1)
    delivery.step(0)

    delivery.step(3)
    delivery.step(0)

    pprint(delivery.observation)



"""
Solving with time limit of 10 seconds
{'objective': 8302569,
 'routes': [{'distance': 1689163,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 2, 'load': 1},
                          {'index': 11, 'load': 10},
                          {'index': 13, 'load': 15},
                          {'index': 0, 'load': 15}]},
            {'distance': 1362162,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 8, 'load': 3},
                          {'index': 9, 'load': 10},
                          {'index': 10, 'load': 14},
                          {'index': 0, 'load': 14}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 1627774,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 7, 'load': 10},
                          {'index': 0, 'load': 10}]},
            {'distance': 2414568,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 5, 'load': 9},
                          {'index': 6, 'load': 10},
                          {'index': 14, 'load': 13},
                          {'index': 12, 'load': 16},
                          {'index': 0, 'load': 16}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 1084634,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 4, 'load': 6},
                          {'index': 1, 'load': 11},
                          {'index': 0, 'load': 11}]},
            {'distance': 124268,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 3, 'load': 10},
                          {'index': 0, 'load': 10}]}],
 'total_distance': 8302569,
 'total_load': 76}
"""