import gym
from gym import spaces
import numpy as np


class CVREnv(gym.Env):
    def __init__(self, capacity, demand, locations):
        super(CVREnv, self).__init__()

        self.capacity = capacity
        self.demand = demand
        self.locations = locations

        self.num_locations = len(locations)
        self.num_vehicles = len(capacity)

        self.action_space = spaces.Discrete(self.num_locations)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_vehicles, self.num_locations + 1), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_state = np.zeros((self.num_vehicles, self.num_locations + 1))
        self.current_capacity = np.copy(self.capacity)
        self.current_demand = np.copy(self.demand)

        return self.current_state

    def step(self, actions):
        assert len(actions) == self.num_vehicles, "Number of actions must match the number of vehicles"

        rewards = np.zeros(self.num_vehicles)
        done = False

        for i, action in enumerate(actions):
            if self.current_capacity[i] >= self.current_demand[action]:
                self.current_state[i][action] = self.current_demand[action]
                self.current_capacity[i] -= self.current_demand[action]
                self.current_demand[action] = 0
                rewards[i] = 1
            else:
                rewards[i] = -1

        if np.sum(self.current_demand) == 0:
            done = True

        return self.current_state, rewards, done, {}

    def render(self, mode='human'):
        print("Current State:")
        print(self.current_state)

    
capacity = [10, 20]  # Capacity of each vehicle
demand = [3, 5, 2, 4]  # Demand at each location
locations = ['A', 'B', 'C', 'D']  # Location names

env = CVREnv(capacity, demand, locations)

state = env.reset()
env.render()

actions = [0, 1]  # Actions for each vehicle (selecting locations)

next_state, rewards, done, _ = env.step(actions)
env.render()

print("Rewards:", rewards)
print("Done:", done)