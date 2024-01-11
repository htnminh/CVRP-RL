import gym
from gymnasium import spaces
import numpy as np

class CVRPEnvironment(gym.Env):
    def __init__(self, capacity, demand, locations):
        super(CVRPEnvironment, self).__init__()
        self.capacity = capacity
        self.demand = demand
        self.locations = locations
        self.steps = 0
        self.max_steps = 100

        self.num_customers = len(demand)
        self.observation_space = spaces.Discrete(self.num_customers + 1)  # Customer index + depot
        self.action_space = spaces.Discrete(self.num_customers)

        self.reset()

    def reset(self):
        self.current_location = 0  # Starting from the depot
        self.remaining_capacity = self.capacity
        self.visited = [False] * self.num_customers
        self.total_reward = 0

    def step(self, action):
        assert self.action_space.contains(action)

        if action == -1 or self.visited[action]:
            reward = 0
        else:
            next_location = self.locations[action]
            distance = self._get_distance(self.current_location, next_location)
            reward = -distance

            self.current_location = next_location
            self.remaining_capacity -= self.demand[action]
            self.visited[action] = True
            self.total_reward += reward

        done = all(self.visited) or self.remaining_capacity <= 0 or self.steps >= self.max_steps
        info = {}

        self.steps += 1

        return action, reward, done, info
    
    def _get_distance(self, location1, location2):
        # Return the distance between two locations (e.g., Euclidean distance)
        return np.linalg.norm(np.array(location1) - np.array(location2))
    

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            # Choose an unvisited customer
            unvisited_customers = [
                action
                for action, visited in enumerate(self.env.visited)
                if not visited and self.env.demand[action] <= self.env.remaining_capacity
            ]
            if unvisited_customers:
                return np.random.choice(unvisited_customers)
            else:
                unvisited_customers = [
                    action
                    for action, visited in enumerate(self.env.visited)
                    if not visited
                ]
                if unvisited_customers:
                    return np.random.choice(unvisited_customers)
                else:
                    return -1  # No unvisited customer
        else:
            return np.argmax(self.q_table[state])


    def update_q_table(self, state, action, reward, next_state):
        next_max = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * next_max - self.q_table[state, action])

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {self.env.total_reward}")

    def test(self):
        state = self.env.reset()
        done = False

        while not done:
            action = np.argmax(self.q_table[state])
            next_state, reward, done, _ = self.env.step(action)
            state = next_state

        print(f"Total Reward: {self.env.total_reward}")


# Create the CVRP environment
capacity = 10
demand = [2, 3, 4, 1, 2]  # Example demand values for customers
locations = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]  # Example locations for customers
env = CVRPEnvironment(capacity, demand, locations)

# Initialize and train the Q-learning agent
agent = QLearningAgent(env)
agent.train(num_episodes=1000)

# Test the trained agent
agent.test()