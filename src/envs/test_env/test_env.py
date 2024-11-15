from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
from utils.dict2namedtuple import convert
import os
#from envs.test_env.generate_maps import *
int_type = np.int16
float_type = np.float32

HEIGHT = 100

# sum reward batch
class TestEnv(MultiAgentEnv):

    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        # Define the agents
        self.n_uavs = 15
        self.n_feats = 2
        self.grid_shape = np.array([20, 30])
        self.x_max, self.y_max = self.grid_shape
        self.env_max = np.asarray(self.grid_shape, dtype=int_type)
        self.state_size = int(self.x_max * self.y_max * self.n_feats + self.n_uavs * 2)


        self.episode_number = 1
        self.observability = getattr(args, "observability", 0)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        maps_dir = os.path.join(current_dir, 'maps')
        map_file = f'map_{self.episode_number}.npy'
        file_path = os.path.join(maps_dir, map_file)
        self.environment_map = np.load(file_path)

        self.steps = 0
        action_labels = {'right': 0, 'down': 1, 'left': 2, 'up': 3, 'weight_increase': 4, 'weight_decrease': 5}
        self.action_move = 4
        self.n_actions = len(action_labels)
        self.grid = np.zeros((self.x_max, self.y_max, self.n_feats), dtype=float_type)
        self.grid[:, :, 0] = self.environment_map

        # 0=events, 1=uavs

        # UAV specific attributes
        self.queue_capacity = 10
        self.float_queues = np.zeros(self.n_uavs, dtype=float_type)
        self.int_queues = np.zeros(self.n_uavs, dtype=int_type)
        self.weights = np.ones(self.n_uavs, dtype=int_type)

        # Base station location and parameters
        self.base_station_pos = np.asarray([10, 15], dtype=int_type)
        self.max_bandwidth = 15

        # Actions (move in four directions and weight adjustment)
        self.actions = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=int_type)
        self.weight_actions = np.array([1, -1], dtype=int_type)  # reduce, increase
        self.action_names = ["right", "down", "left", "up", "weight_increase", "weight_decrease"]

        # Episode and reward settings
        self.episode_limit = getattr(args, "episode_limit", 100)
        self.time_reward = getattr(args, "reward_time", -0.1)
        self.send_to_base_reward = getattr(args, "reward_base", 10.0)

        # Initialize the internal state
        self.uavs = np.zeros((self.n_uavs, 2), dtype=int_type)
        self.steps = 0
        self.sum_rewards = 0

        # Initialize the knowledge map
        self.knowledge_map = np.full((self.x_max, self.y_max), 2, dtype=int_type)  # Start with all cells as unvisited (2)
        self.observed_state = np.full((self.x_max, self.y_max), 2, dtype=int_type)  # Start with all cells as unvisited (2)
        self._place_actors(self.uavs, 1)


        self.cell_distance = 20
        self.step_in_sec = 3
        self.packet_size = 200*1024
        
        self.obs_size = self.get_obs_size()
        self.cum_reward = []

    def reset(self):
        # Reset the knowledge map
        self.knowledge_map = np.full((self.x_max, self.y_max), 2, dtype=int_type)  # Unvisited cells
        self.observed_state = np.full((self.x_max, self.y_max), 2, dtype=int_type)  # Start with all cells as unvisited (2)
        self.int_queues.fill(0)
        self.float_queues.fill(0)
        self.weights.fill(1)
        self.steps = 0
        self.sum_rewards = 0
        self.uavs = np.zeros((self.n_uavs, 2), dtype=int_type)
        self.grid.fill(0.0)
        self._place_actors(self.uavs, 1)
        rows, cols = self.grid_shape[0], self.grid_shape[1]
        ratio_of_ones = 0.7  # Example ratio
        current_dir = os.path.dirname(os.path.abspath(__file__))
        maps_dir = os.path.join(current_dir, 'maps')
        map_file = f'map_{self.episode_number}.npy'
        file_path = os.path.join(maps_dir, map_file)
        self.environment_map = np.load(file_path)
        self.grid[:, :, 0] = self.environment_map
        print('Episode', self.episode_number)
        self.episode_number += 1
        self.cum_reward = []


        return self.get_obs(), self.get_state()

    def step(self, actions):
        reward = self.time_reward
        terminated = False
        print(actions)
        print(np.count_nonzero(self.grid[:, :, 0]))
        #print(self.grid[:, :, 1])
        # Move the UAVs and handle events
        for u in range(self.n_uavs):
            if actions[u] >= self.action_move:
                # Adjust weights (if applicable)
                new_weight = self.weights[u] + self.weight_actions[actions[u] - self.action_move]
                self.weights[u] = np.clip(new_weight, 0, 5)
            else:
                # Move UAV
                new_pos, collide = self._move_actor(self.uavs[u, :], actions[u])
                if not collide:
                    self.grid[self.uavs[u, 0], self.uavs[u, 1], 1] = 0
                    self.grid[new_pos[0], new_pos[1], 1] = u + 1
                    self.uavs[u, :] = new_pos

                    x, y = new_pos[0], new_pos[1]

                    if self.grid[x, y, 0]:
                        self.observed_state[x, y] = 1
                        if self.float_queues[u] < self.queue_capacity:
                            self.float_queues[u] += 1
                            self.int_queues[u] = int(self.float_queues[u])
                            self.grid[x, y, 0] = 0  
                            self.observed_state[x, y] = 0
                        if self.knowledge_map[x,y] == 2:
                            self.knowledge_map[x, y] = 1
                    else:
                        if self.knowledge_map[x, y] == 2:
                            self.knowledge_map[x, y] = 0
                        self.observed_state[x, y] = 0
                    if self.observability:
                        self.check_position(x, y+1)
                        self.check_position(x, y-1)
                        self.check_position(x-1, y)
                        self.check_position(x+1, y)
        # All UAVs share the updated knowledge map

        # Distribute bandwidth based on weights
        total_weight = np.sum(self.weights)
        if total_weight == 0:
            self.weights.fill(1)
            total_weight = np.sum(self.weights)
        bandwidths = (self.weights / total_weight) * self.max_bandwidth

        for u in range(self.n_uavs):
            # Calculate the UAV's distance to the base station
            distance_to_base_2D = np.linalg.norm(self.uavs[u, :]+0.5  - self.base_station_pos[:2])*self.cell_distance
            distance_to_base_3D = np.sqrt(distance_to_base_2D ** 2 + HEIGHT ** 2)

            # Calculate data rate based on the distributed bandwidth and distance
            data_rate = self.calculate_data_rate(bandwidths[u], distance_to_base_3D)

            # Simulate sending data to base station
            if self.float_queues[u] > 0:

                # Store the previous integer size for comparison
                previous_int_size = int(self.float_queues[u])

                # Calculate how much data is transmitted in this step
                transmitted_data = min(data_rate*self.step_in_sec/self.packet_size, self.float_queues[u])

                # Decrease the queue by the transmitted data amount
                self.float_queues[u] -= transmitted_data

                # Check if the integer magnitude has changed
                current_int_size = int(self.float_queues[u])
                if current_int_size < previous_int_size:
                    completed_packets = previous_int_size - current_int_size
                    reward += self.send_to_base_reward * completed_packets  # Reward for integer magnitude change

                # If the queue is emptied, give the reward for completing the entire packet
                if self.float_queues[u] <= 0:
                    self.float_queues[u] = 0  # Ensure the queue is fully cleared
                    reward += self.send_to_base_reward  # Reward for completing the entire transmission
                self.int_queues[u] = int(self.float_queues[u])

        self.sum_rewards += reward
        self.cum_reward.append(self.sum_rewards)
        self.steps += 1
        if self.steps >= self.episode_limit:
            terminated = True

        return reward, terminated, {}
    
    def get_knowledge_map(self):
        return self.knowledge_map
    
    def get_obs_agent(self,agent_id):
        # Since all UAVs share the knowledge map, each UAV's observation includes the shared map
        # along with any additional information (e.g., own queue, weight), and all UAV positions

        # Flatten the knowledge map
        observed_state_flat = self.observed_state.flatten()

        # Get all UAV positions
        uav_positions = self.grid[:, :, 1].flatten()   

        flattened_queues = self.int_queues.flatten()
        flattened_weights = self.weights.flatten()
        position = np.zeros(self.grid_shape)
        position[self.uavs[agent_id, 0], self.uavs[agent_id, 1]] = 1.0

        # Concatenate the knowledge map with agent-specific information and UAV positions
        concatenated_array = np.concatenate((observed_state_flat, uav_positions, flattened_queues, flattened_weights,position.flatten()))

        return concatenated_array

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_uavs)]
        return agents_obs

    def get_full_obs_agent(self, agent_id, batch=0):

        flattened_grid = self.grid[batch,:,:,:].flatten()  
        flattened_queues = self.int_queues[:,batch].flatten()
        flattened_weights = self.weights[:,batch].flatten()
        position = np.zeros(self.grid_shape)
        position[self.uavs[agent_id, 0], self.uavs[agent_id, 1]] = 1.0
        concatenated_array = np.concatenate((flattened_grid, flattened_queues, flattened_weights,position.flatten()))
        return concatenated_array 

    def get_full_obs(self):
        agents_obs = [self.get_full_obs_agent(i) for i in range(self.n_uavs)]
        return agents_obs
    
    def get_obs_size(self):
        # Observation size is the size of the knowledge map plus all UAV positions and agent-specific info
        return int(self.x_max * self.y_max * self.n_feats + self.n_uavs*2 + self.grid_shape[0] * self.grid_shape[1])  

    def get_state(self):
        # The global state could include the full environment map and UAV positions
        flattened_grid = self.grid[:, :, :].flatten()  
        flattened_queues = self.int_queues.flatten()
        flattened_weights = self.weights.flatten()
        concatenated_array = np.concatenate((flattened_grid, flattened_queues, flattened_weights))
        return concatenated_array

    def get_state_size(self):
        # State size is the size of the environment map plus UAV positions
        return int(self.x_max * self.y_max * self.n_feats + self.n_uavs*2)

    def calculate_data_rate(self, bandwidth, distance):
        if bandwidth == 0:
            return 0
        p_n = 0.1  # Transmit power (in watts)
        alpha_u = 1  # Channel power gain at reference distance
        B = 10**5  # Total bandwidth (in Hz)
        N_0 = 10**(-12.7)*alpha_u  # Noise power spectral density (in watts/Hz)
	# Calculate the path loss (g_n_u_O)
        g_n_u_O = alpha_u / distance**2
    
	# Calculate the SNR (gamma_n_u_O)
        gamma_n_u_O = (p_n * g_n_u_O) / (B * bandwidth * N_0)
    
	# Calculate the channel capacity using the Shannon-Hartley theorem
        C_n_u_O = B * bandwidth * np.log2(1 + gamma_n_u_O)
        #print(C_n_u_O,gamma_n_u_O,bandwidth)
    
        return C_n_u_O

    def get_avail_actions(self):
        avail_actions = []
        for uav_id in range(self.n_uavs):
            avail_uav = self.get_avail_agent_actions(uav_id)
            avail_actions.append(avail_uav)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def render(self):
        raise NotImplementedError

    def _place_actors(self, actors, type_id):
        for a in range(actors.shape[0]):
            is_free = False
            while not is_free:
                actors[a, 0] = np.random.randint(self.env_max[0])
                actors[a, 1] = np.random.randint(self.env_max[1])
                is_free = self.grid[actors[a, 0], actors[a, 1], type_id] == 0
            self.grid[actors[a, 0], actors[a, 1], type_id] = a + 1
            self.uavs[a, :] = actors[a, :]
            if self.grid[actors[a, 0], actors[a, 1], 0]:
                self.observed_state[actors[a, 0], actors[a, 1]] = 1
                if self.float_queues[a] < self.queue_capacity:
                    self.float_queues[a] += 1
                    self.int_queues[a] = int(self.float_queues[a])
                    self.grid[actors[a, 0], actors[a, 1], 0] = 0  
                    self.observed_state[actors[a, 0], actors[a, 1]] = 0
                self.knowledge_map[actors[a, 0], actors[a, 1]] = 1 
            else:
                if self.knowledge_map[actors[a, 0], actors[a, 1]] == 2:
                    self.knowledge_map[actors[a, 0], actors[a, 1]] = self.environment_map[actors[a, 0], actors[a, 1]]
                self.observed_state[actors[a, 0], actors[a, 1]] = 0
            if self.observability:
                self.check_position(actors[a, 0], actors[a, 1]+1)
                self.check_position(actors[a, 0], actors[a, 1]-1)
                self.check_position(actors[a, 0]-1, actors[a, 1])
                self.check_position(actors[a, 0]+1, actors[a, 1])


    def check_position(self,x,y):
        if x >= 0 and x < self.grid_shape[0] and y >= 0 and y < self.grid_shape[1]:    
            self.observed_state[x,y] = self.grid[x, y, 0]
            if self.knowledge_map[x, y] == 2:
                self.knowledge_map[x, y] = self.environment_map[x,y]


    def _move_actor(self, pos, action):
        new_pos = self._env_bounds(pos + self.actions[action])
        collision_mask = [1]
        collision = np.sum(self.grid[new_pos[0], new_pos[1], collision_mask]) > 0
        return new_pos, collision

    def _env_bounds(self, positions):
        positions = np.minimum(positions, self.env_max - 1)
        positions = np.maximum(positions, 0)
        return positions

    def close(self):
        print("Closing UAV Environment")

    def seed(self):
        raise NotImplementedError

    def get_stats(self):
        pass

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_uavs,
                    "episode_limit": self.episode_limit}
        return env_info
