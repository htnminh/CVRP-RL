from pprint import pprint, pformat
from typing import Any, SupportsFloat
import time
import warnings

import numpy as np
from numpy.random import randint

from scipy.spatial.distance import cdist, euclidean


# class Delivery(gymnasium.Env):
class Delivery:
    def __init__(self, n_stops=10, max_demand=10, max_vehicle_cap=30, max_env_size=1_000_000,
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


    def reset(self, seed=None):
        try:
            self.pre_reset_observation = self.observation
            self.pre_reset_info = self.info   
        except AttributeError:
            self.pre_reset_observation = None
            self.pre_reset_info = None
        
        # this seed is used to fix compatibility problems between gymnasium and stable-baselines3
        # if seed is not None:
        #     super().reset(seed=seed)
        # else:
        #     super().reset()

        info = dict()

        # self.action_space = spaces.Discrete(self.n_stops, seed=self.gym_seed)        
        # self.observation_space = spaces.Dict({
        #     'coord': spaces.Box(
        #         low=0, high=self.max_env_size,
        #         shape=(self.n_stops, 2), dtype=int,
        #         seed=self.gym_seed),
        #     'demand': spaces.Box(
        #         low=0, high=self.max_demand,
        #         shape=(self.n_stops, ), dtype=int,
        #         seed=self.gym_seed),
        #     'visited': spaces.MultiBinary(self.n_stops, seed=self.gym_seed),
        #         # visited of stop 0 should always be 0
        #     'current_load': spaces.Box(
        #         low=0, high=self.max_vehicle_cap,
        #         shape=(1, ), dtype=int,
        #         seed=self.gym_seed),
        #     'current_stop': spaces.Discrete(self.n_stops, seed=self.gym_seed),
        #     'current_length': spaces.Box(
        #         low=0, high=np.inf,
        #         shape=(1, ), dtype=int,
        #         seed=self.gym_seed)
        # })
        # # self.reward_range = (-np.inf, 0)
        # self.reward_range = (-np.inf, np.inf)

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
        # assert self.action_space.contains(action), f"Invalid action {action}"
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


    def epsilon_greedy(self, epsilon, q_table):
        if np.random.random() < epsilon:
            # print('random:', self.observation['visited'])
            res = np.random.choice(
                np.arange(0, self.n_stops, 1)[self.observation['visited'] == 0]
            )
            # print(res)
            return res
        else:
            q_table_copy = q_table.copy()
            q_table_copy[:, self.observation['visited'] == 1] = - np.inf
            return np.argmax(q_table_copy[self.observation['current_stop'], :])


if __name__ == '__main__':
    delivery = Delivery(n_stops=4, max_demand=10, max_vehicle_cap=30, max_env_size=1_000,
                        gen_seed=0, gym_seed=0, print_input=True, print_terminated=True)
    
    N_STEPS = 10_000
    EPSILON = 0.5 
    ALPHA = 0.1  # learning rate
    GAMMA = 0.5  # discount factor
    
    # initialize q_table
    q_table_original = - cdist(delivery.other_data['stops_coords'], delivery.other_data['stops_coords'])
    np.fill_diagonal(q_table_original, - 2 * delivery.max_env_size * delivery.n_stops)
    q_table = q_table_original.copy()
    print(q_table)

    # train
    for step in range(N_STEPS):
        current_stop = delivery.observation['current_stop']
        action = delivery.epsilon_greedy(EPSILON, q_table)
        observation, reward, terminated, truncated, info = delivery.step(action)
        # print(f'update at {current_stop, action}')
        q_table[current_stop, action] = \
            (1 - ALPHA) * q_table[current_stop, action] + \
                ALPHA * (reward + GAMMA * np.max(q_table[action, :]))
        if terminated or truncated:
            delivery.reset()
    
    print()
    print('Original q_table:')
    print(q_table_original)
    
    print()
    print(f'Trained q_table with N_STEPS={N_STEPS}, EPSILON={EPSILON}, ALPHA={ALPHA}, GAMMA={GAMMA}:')
    print(q_table)

