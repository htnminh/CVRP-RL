from pprint import pprint, pformat
from typing import Any, SupportsFloat
import time
import warnings

import tqdm

import numpy as np
from numpy.random import randint

from scipy.spatial.distance import cdist, euclidean

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import gymnasium
from gymnasium import spaces

from stable_baselines3 import DQN, A2C, PPO

import torch
from torch import tensor



class Delivery(gymnasium.Env):
    def __init__(self, n_stops=10, max_demand=10, max_vehicle_cap=30, max_env_size=1_0000_000,
                 gen_seed=None, gym_seed=None, print_input=True, print_terminated=True) -> None:
        # Print input info
        if print_input:
            print('n_stops:', n_stops)
            print('max_demand:', max_demand)
            print('max_vehicle_cap:', max_vehicle_cap)
            print('max_env_size:', max_env_size)
            print('gen_seed:', gen_seed)
            print('gym_seed:', gym_seed)

        # Data generation
        assert n_stops > 1 and max_demand > 0 and max_vehicle_cap > 0 and max_env_size > 0 
        self.n_stops = n_stops
        self.max_demand = max_demand
        self.max_vehicle_cap = max_vehicle_cap
        self.max_env_size = max_env_size
        self.gen_seed = gen_seed
        self.gym_seed = gym_seed
        assert self.max_demand <= self.max_vehicle_cap
        self.print_terminated = print_terminated
        
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
            ax1.annotate(f'{route_distance} [{route_load}]',
                         ((route_coords[-1, 0] + route_coords[-2, 0])/2, (route_coords[-1, 1] + route_coords[-2, 1])/2),
                         fontsize=8, textcoords="offset points", xytext=(0, 0), ha='center', color=_color)

        if show:
            plt.show()


    def reset(self, seed=None):
        try:
            self.pre_reset_observation = self.observation
            self.pre_reset_info = self.info   
        except AttributeError:
            self.pre_reset_observation = None
            self.pre_reset_info = None
        
        # this seed is used to fix compatibility problems between gymnasium and stable-baselines3
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset()

        info = dict()

        self.action_space = spaces.Discrete(self.n_stops, seed=self.gym_seed)        
        self.observation_space = spaces.Dict({
            'coord': spaces.Box(
                low=0, high=self.max_env_size,
                shape=(self.n_stops, 2), dtype=int,
                seed=self.gym_seed),
            'demand': spaces.Box(
                low=0, high=self.max_demand,
                shape=(self.n_stops, ), dtype=int,
                seed=self.gym_seed),
            'visited': spaces.MultiBinary(self.n_stops, seed=self.gym_seed),
                # visited of stop 0 should always be 0
            'current_load': spaces.Box(
                low=0, high=self.max_vehicle_cap,
                shape=(1, ), dtype=int,
                seed=self.gym_seed),
            'current_stop': spaces.Discrete(self.n_stops, seed=self.gym_seed),
            'current_length': spaces.Box(
                low=0, high=np.inf,
                shape=(1, ), dtype=int,
                seed=self.gym_seed)
        })
        # self.reward_range = (-np.inf, 0)
        self.reward_range = (-np.inf, np.inf)

        observation = dict(
            coord=self.other_data['stops_coords'].astype(int),
            demand=self.ortools_data['demands'].astype(int),
            visited=np.zeros(self.n_stops).astype(int),
            current_load=np.array([self.ortools_data['vehicle_caps'][0], ], dtype=int),
            current_stop=0,
            current_length=np.array([0, ], dtype=int)
        )
        info['n_routes_redundant'] = 0
        info['full_solution'] = []
        info['ortools_format_solution'] = dict(
            objective=0,
            routes=[
                dict(
                    distance=0,
                    segments=[dict(index=0, load=0), ]
                )
            ],
            total_distance=0,
            total_load=0
        )
        info['current_route_length'] = 0
        info['total_reward'] = 0

        self.observation = observation
        self.info = info
        
        
        return observation, info
    

    def step(self, action):
        """
        Note that the 'load' used in ortools is different from the 'load' used in gymnasium
        - load of ortools is the total load that the vehicle has served
        - load of gymnasium is the remaining load that the vehicle can serve
            = vehicle_cap - load of ortools
        except data in the ortools_format_solution
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.info['n_routes_redundant'] += 1

        INVALID_REWARD = - 2 * self.n_stops
        # INVALID_REWARD = 2 * self.n_stops * self.max_env_size
 
        if self.observation['visited'][action] != 0:
            # print(f"Stop {action} has already been visited")
            # self.observation['current_length'] += 2 * self.max_env_size
            reward = INVALID_REWARD
            self.info['total_reward'] += reward
            terminated = False
        elif self.observation['current_load'] < self.observation['demand'][action]:
            # print(f"Current load cannot be negative")
            # self.observation['current_length'] += 2 * self.max_env_size
            reward = INVALID_REWARD
            self.info['total_reward'] += reward
            terminated = False
        elif self.observation['current_stop'] == 0 and action == 0:
            # print(f"Cannot stay in depot")
            # self.observation['current_length'] += 2 * self.max_env_size
            reward = INVALID_REWARD
            self.info['total_reward'] += reward
            terminated = False
        else:
            # move to next stop
            # pprint(self.observation)
            # pprint(self.info)
            segment_length = int(euclidean(
                self.observation['coord'][self.observation['current_stop']], self.observation['coord'][action]))
            self.observation['current_length'] += segment_length
            if action != 0:
                self.observation['visited'][action] = 1
            self.observation['current_load'] -= self.observation['demand'][action]
            self.observation['current_stop'] = action
            self.info['full_solution'].append(action)
            self.info['ortools_format_solution']['objective'] += segment_length
            self.info['ortools_format_solution']['total_distance'] += segment_length
            self.info['ortools_format_solution']['total_load'] += self.observation['demand'][action]
            self.info['current_route_length'] += segment_length
            self.info['ortools_format_solution']['routes'][-1]['distance'] += segment_length
            self.info['ortools_format_solution']['routes'][-1]['segments'].append(dict(
                index=int(action),
                load=self.ortools_data['vehicle_caps'][0] - int(self.observation['current_load'])
            ))

            # reset if back to depot
            if self.observation['current_stop'] == 0:
                self.observation['current_load'] = np.array([self.ortools_data['vehicle_caps'][0], ], dtype=int)
                self.info['current_route_length'] = 0
                self.info['ortools_format_solution']['routes'].append(
                    dict(
                        distance=0,
                        segments=[dict(index=0, load=0), ]
                    )
                )

            # calc reward
            reward = -segment_length / self.max_env_size
            self.info['total_reward'] += reward
            # reward = segment_length

            # check if done
            if np.sum(self.observation['visited']) == self.n_stops - 1 and self.observation['current_stop'] == 0:
                if self.print_terminated:
                    print(f'TERMINATED:   length={self.observation["current_length"]}\t moves={self.info["n_routes_redundant"]}\t total_reward={self.info["total_reward"]}')
                    print('\t', [int(action) for action in self.info['full_solution']])
                terminated = True
                self.last_solution_info = self.info.copy()  # may raise an attribute error if no solution is found, which is rarely the case
                self.reset()
            else:
                terminated = False
        

        # check and print observation belongs to spaces in detail of each key
        # for key in self.observation.keys():
        #     assert self.observation_space[key].contains(self.observation[key]), f"Invalid observation {key}: {self.observation[key]}"

        # # check and print reward belongs to reward range
        # assert self.reward_range[0] <= reward <= self.reward_range[1], f"Invalid reward {reward}"

        return (
            self.observation,
            reward,
            terminated,
            False, # truncated (bool) – Whether the truncation condition outside the scope of the MDP is satisfied.
            self.info
        )
    

    @staticmethod
    def convert_observation_to_tensor(observation: dict):
        res = observation.copy()
        for key in res.keys():
            res[key] = tensor(res[key])
        return res


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

    """ 
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
    del delivery
    """


    """
    delivery = Delivery(n_stops=15)
    # model = DQN("MlpPolicy", delivery, verbose=1)
    model = DQN("MultiInputPolicy", delivery, verbose=1)
    model.learn(total_timesteps=100, log_interval=4)
    model.save("dqn_delivery")
    model = DQN.load("dqn_delivery")
    observation, info = delivery.reset()
    while True:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = delivery.step(action)
        if terminated or truncated:
            break
    """
    # initialization
    N_STOPS = 10
    N_MOVE_THRESHOLD = N_STOPS * 100
    TOTAL_TIMESTEPS = 500

    print('='*25 + 'Initialization' + '='*25)
    delivery = Delivery(n_stops=N_STOPS, gen_seed=20, gym_seed=20, max_env_size=1000)
    
    # quick warning
    warnings.warn(
        """Sometimes an algorithm other than OR-Tools, typically PPO, may not be able to give a solution or take too long do so. """
        """In that case, the solution will be fall back to its last found solution during training."""
    )

    # ortools
    print('='*25 + 'OR-Tools' + '='*25)
    pprint(ortools_solution := delivery.get_ortools_solution(time_limit_seconds=10))
    delivery.visualize_ortools_solution(solution=ortools_solution)

    """
Solving with time limit of 10 seconds
{'objective': 7104,
 'routes': [{'distance': 1459,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 4, 'load': 6},
                          {'index': 3, 'load': 16},
                          {'index': 0, 'load': 16}]},
            {'distance': 1654,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 9, 'load': 7},
                          {'index': 1, 'load': 12},
                          {'index': 10, 'load': 16},
                          {'index': 0, 'load': 16}]},
            {'distance': 1340,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 2, 'load': 1},
                          {'index': 5, 'load': 10},
                          {'index': 12, 'load': 13},
                          {'index': 0, 'load': 13}]},
            {'distance': 1214,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 14, 'load': 3},
                          {'index': 8, 'load': 6},
                          {'index': 7, 'load': 16},
                          {'index': 0, 'load': 16}]},
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
            {'distance': 1437,
             'segments': [{'index': 0, 'load': 0},
                          {'index': 6, 'load': 1},
                          {'index': 13, 'load': 6},
                          {'index': 11, 'load': 15},
                          {'index': 0, 'load': 15}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]},
            {'distance': 0,
             'segments': [{'index': 0, 'load': 0}, {'index': 0, 'load': 0}]}],
 'total_distance': 7104,
 'total_load': 76}
    """

    # a2c
    print('='*25 + 'A2C' + '='*25)
    observation, info = delivery.reset()
    model = A2C("MultiInputPolicy", delivery, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    last_solution_info = delivery.last_solution_info.copy()
    model.save("a2c_delivery")
    model = A2C.load("a2c_delivery")
    
    fallback = False
    observation, info = delivery.reset()
    while True:
        action, _states = model.predict(observation, deterministic=False)
        observation, reward, terminated, truncated, info = delivery.step(action)
        # pprint(delivery.observation)
        if terminated or truncated:
            break
        elif info["n_routes_redundant"] > N_MOVE_THRESHOLD:
            warnings.warn(f'The algorithm took too long to give a solution, '
                           'falling back to the last found solution during training.')
            fallback = True
            break

    if not fallback:
        final_info = delivery.pre_reset_info.copy()
    else:
        final_info = last_solution_info.copy()
    print([int(action) for action in final_info['full_solution']])
    pprint(final_info['ortools_format_solution'])
    delivery.visualize_ortools_solution(solution=final_info['ortools_format_solution'])
    input('Enter to continue...')


    # dqn
    print('='*25 + 'DQN' + '='*25)
    observation, info = delivery.reset()
    model = DQN("MultiInputPolicy", delivery, verbose=1, learning_rate=0.001)
    # model = DQN("MultiInputPolicy", delivery, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    last_solution_info = delivery.last_solution_info.copy()
    model.save("dqn_delivery")
    model = DQN.load("dqn_delivery")

    fallback = False
    observation, info = delivery.reset()
    while True:
        action, _states = model.predict(observation, deterministic=False)
        observation, reward, terminated, truncated, info = delivery.step(action)
        # pprint(delivery.observation)
        if terminated or truncated:
            break
        elif info["n_routes_redundant"] > N_MOVE_THRESHOLD:
            warnings.warn(f'The algorithm took too long to give a solution, '
                           'falling back to the last found solution during training.')
            fallback = True
            break

    if not fallback:
        final_info = delivery.pre_reset_info.copy()
    else:
        final_info = last_solution_info.copy()
    print([int(action) for action in final_info['full_solution']])
    pprint(final_info['ortools_format_solution'])
    delivery.visualize_ortools_solution(solution=final_info['ortools_format_solution'])
    input('Enter to continue...')


    # ppo
    # sometimes ppo does not give a solution at all
    print('='*25 + 'PPO' + '='*25)
    observation, info = delivery.reset()
    model = PPO("MultiInputPolicy", delivery, verbose=1, learning_rate=0.001)
    # model = PPO("MultiInputPolicy", delivery, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    last_solution_info = delivery.last_solution_info.copy()
    model.save("ppo_delivery")
    model = PPO.load("ppo_delivery")

    fallback = False
    observation, info = delivery.reset()
    while True:
        action, _states = model.predict(observation, deterministic=False)
        observation, reward, terminated, truncated, info = delivery.step(action)
        # pprint(delivery.observation)
        if terminated or truncated:
            break
        elif info["n_routes_redundant"] > N_MOVE_THRESHOLD:
            warnings.warn(f'The algorithm took too long to give a solution, '
                           'falling back to the last found solution during training.')
            fallback = True
            break

    if not fallback:
        final_info = delivery.pre_reset_info.copy()
    else:
        final_info = last_solution_info.copy()
    print([int(action) for action in final_info['full_solution']])
    pprint(final_info['ortools_format_solution'])
    delivery.visualize_ortools_solution(solution=final_info['ortools_format_solution'])
    input('Enter to continue...')


    # sac, td3, ddpg: The algorithm only supports (<class 'gymnasium.spaces.box.Box'>,) as action spaces but Discrete(15) was provided
    # this could be by-passed easily by converting the action to one-hot vector or an integer but it is not worth trying
    # since the algorithms are simply not designed for discrete action spaces

